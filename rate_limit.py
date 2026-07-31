"""A rolling-window rate limiter, kept free of any web framework.

Answers one question: has this caller made more than N requests in the last W
seconds? The window is a true rolling one -- every request is remembered by its
timestamp and forgotten W seconds later, so there is no fixed-bucket edge where a
caller gets 2N requests by straddling the boundary.

    limiter = RateLimiter(limit=20, window_seconds=3600)
    try:
        remaining = limiter.check(client_ip(xff_header, fallback))
    except RateLimited as e:
        ...  # answer 429, and tell them e.retry_after seconds

Nothing here imports FastAPI, so abi-server can take this module as-is; mapping
RateLimited onto an HTTP response is the caller's job, because the two services
shape their error bodies differently.

Two things this deliberately is not. It is **per process**: the counters live in
memory, so a restart clears them and N App Runner instances mean N times the
limit. And it is **not a security control** -- the key is derived from
X-Forwarded-For, which the caller can set to anything. It curbs casual hammering
and runaway scripts, which is what it is for. A limit that must hold exactly, or
against someone deliberately evading it, needs shared storage and an identity the
client cannot choose.
"""
from collections import deque
import math
import threading
import time


class RateLimited(Exception):
    """The caller has used up their allowance and should come back later.

    `retry_after` is whole seconds until the oldest request in the window expires,
    which is the earliest moment the next one can succeed -- suitable for the
    Retry-After header as-is. `key` is carried for logging; it is a client IP, so
    think before putting it in a response body.
    """

    def __init__(self, key, limit, window_seconds, retry_after):
        super().__init__(
            f"{limit} requests per {window_seconds}s exceeded; "
            f"retry in {retry_after}s")
        self.key = key
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after = retry_after


def client_ip(forwarded_for, fallback=None):
    """The client-most address in an X-Forwarded-For header, or `fallback`.

    Proxies append to this header left to right, so the left-most entry is the
    original client and everything after it is infrastructure. Behind App Runner
    the socket peer (`request.client.host`) is the load balancer, identical for
    every caller -- keying on that would put the whole world in one bucket and
    ration the entire service to `limit` requests an hour. Hence this.

    The value is caller-supplied and trivially spoofed; see the module docstring.
    `fallback` is used when the header is missing or empty, which is the normal
    case for a direct connection in local development.
    """
    if forwarded_for:
        # "client, proxy1, proxy2" -- take the first non-empty entry. Quotes and
        # stray whitespace show up in the wild; a port suffix does not, on any of
        # the proxies in front of these services, so it is left alone rather than
        # mis-parsed (stripping ":443" would corrupt a bare IPv6 address).
        for part in forwarded_for.split(","):
            part = part.strip().strip('"')
            if part:
                return part
    return fallback


class RateLimiter:
    """At most `limit` requests per key per `window_seconds`, counted in memory.

    Thread-safe: FastAPI runs non-async dependencies in a worker thread, and two
    requests from one caller can land at once, so check-and-record has to be one
    atomic step or a caller can slip past the limit by racing.

    `now` is injectable so the behaviour can be tested without sleeping through a
    real window. It defaults to a monotonic clock rather than wall time, so the
    limiter is unaffected by NTP steps or daylight-saving changes.
    """

    def __init__(self, limit, window_seconds, now=time.monotonic):
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        if window_seconds <= 0:
            raise ValueError(
                f"window_seconds must be positive, got {window_seconds}")
        self.limit = limit
        self.window_seconds = window_seconds
        self._now = now
        self._hits = {}                  # key -> deque of timestamps, oldest first
        self._lock = threading.Lock()
        self._next_sweep = now() + window_seconds

    def check(self, key):
        """Record one request against `key`; return how many remain in the window.

        Raises RateLimited instead, without recording anything, when the key is at
        its limit -- a refused request must not extend the window, or a caller who
        keeps hammering after a 429 would never be let back in.
        """
        now = self._now()
        with self._lock:
            if now >= self._next_sweep:
                self._sweep(now)

            hits = self._hits.get(key)
            if hits is None:
                hits = self._hits[key] = deque()

            cutoff = now - self.window_seconds
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.limit:
                # hits[0] is the oldest request still counting; once it ages out
                # there is room for one more.
                retry_after = max(1, math.ceil(hits[0] + self.window_seconds - now))
                raise RateLimited(key, self.limit, self.window_seconds, retry_after)

            hits.append(now)
            return self.limit - len(hits)

    def _sweep(self, now):
        """Drop keys with nothing left in the window. Caller holds the lock.

        Without this, one dict entry accumulates per address ever seen and the
        process leaks slowly for as long as it runs. Sweeping once per window is
        enough -- a key is only worth keeping while it has a live timestamp -- and
        keeps the cost off the common path.
        """
        cutoff = now - self.window_seconds
        stale = [k for k, hits in self._hits.items() if not hits or hits[-1] <= cutoff]
        for k in stale:
            del self._hits[k]
        self._next_sweep = now + self.window_seconds

    def describe(self):
        return f"{self.limit} requests per {self.window_seconds}s per client"
