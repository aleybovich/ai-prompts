# Nimbus Notes API

The Nimbus API lets you read and write notes programmatically. It is available on
Pro and Team plans. The Free plan does not include API access.

## Authentication

Create an API key in Settings > Developer > API keys. Send it as a bearer token
in the `Authorization` header:

```
Authorization: Bearer nk_live_xxxxxxxxxxxxxxxx
```

Keys starting with `nk_live_` work against your real account. Keys starting with
`nk_test_` work against an isolated sandbox account with fake data, which is
useful for development.

## Base URL and versioning

The base URL is `https://api.nimbusnotes.com/v2`. The API is versioned in the
path. Version 1 (`/v1`) is deprecated and will be shut down; new integrations
should use `/v2`.

## Rate limits

The API allows 120 requests per minute per API key on Pro, and 600 requests per
minute per key on Team. If you exceed the limit, the API returns HTTP status
`429 Too Many Requests` with a `Retry-After` header telling you how many seconds
to wait. Back off and retry after that delay.

## Errors

Errors return a JSON body with a machine-readable `code` and a human-readable
`message`. Common codes:

- `NIMBUS_AUTH_401` — the API key is missing, malformed, or revoked.
- `NIMBUS_FORBIDDEN_403` — the key is valid but not allowed to access that note
  (for example, a note in another user's account).
- `NIMBUS_NOT_FOUND_404` — no note exists with that ID.
- `NIMBUS_RATE_LIMIT_429` — you hit the rate limit; see the `Retry-After` header.
- `NIMBUS_PAYLOAD_413` — the request body is larger than the 10 MB limit.

## Webhooks

You can register a webhook URL to be notified when notes change. Nimbus signs each
webhook request with an HMAC signature in the `X-Nimbus-Signature` header so you
can verify it really came from Nimbus.
