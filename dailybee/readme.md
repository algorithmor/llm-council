# 🐝 DailyBee

**DailyBee** gives you a calm daily bulletin of what's new on YouTube. It reads your
subscriptions and shows every video posted in the last 24 hours, grouped by channel,
with a short summary of each. There's no algorithmic feed and no autoplay rabbit hole.
If you want to watch something, click **Watch here** to play it in a pop-up player, or
open it on YouTube.

DailyBee is a fork of [yarr](https://github.com/nkanaev/yarr), a small self-hosted
RSS reader. The full reader is still there at `/reader` if you want it.

## Features

- **Daily bulletin**: everything from the last 24 hours (or 2 or 7 days), grouped by
  channel. It shows the total count and watch time, video lengths, view counts, and
  badges for Shorts and live streams.
- **Watch inline or on YouTube**: the pop-up player uses youtube-nocookie.com. Videos
  you open are marked as watched.
- **Mark watched / save for later**, "mark all as watched", and "hide watched".
- **Hide Shorts** if you only want full-length videos.
- **Syncs your subscriptions automatically** once a day, so new channels show up by
  themselves. It can also remove channels you unsubscribe from.
- **Uses very little API quota**: new uploads come from each channel's free public RSS
  feed. The API key is only used to list your subscriptions (1 unit per 50 channels)
  and look up video lengths (1 unit per 50 videos). The free quota is 10,000 units a day.
- **Works with private subscriptions too**: import Google Takeout's
  `subscriptions.csv`, or add channels by URL in the reader.
- Light and dark mode, works on phones, optional login, and runs as a single container.

## Quick start (Docker or Podman)

```sh
cd dailybee
cp .env.example .env        # optional: put your API key and channel here
docker compose up -d --build
# or
podman compose up -d --build     # or: podman-compose up -d --build
```

Open <http://localhost:7070> and follow the **Setup** page. Your data lives in the
`dailybee-data` volume.

Without Compose:

```sh
docker build -t localhost/dailybee .            # or: podman build -t localhost/dailybee .
docker run -d --name dailybee -p 127.0.0.1:7070:7070 \
  -v dailybee-data:/data \
  -e DAILYBEE_YOUTUBE_API_KEY=AIza... \
  -e DAILYBEE_YOUTUBE_CHANNEL=@yourhandle \
  -e TZ=Europe/London \
  localhost/dailybee                              # use `podman run` the same way
```

To start it at boot with Podman, use the Quadlet unit in
[`deploy/dailybee.container`](deploy/dailybee.container).

> **Bind mounts with rootless Podman:** the container runs as uid 1000. If you mount a
> host directory instead of a named volume, add `--userns=keep-id` (or `chown` the
> directory) and `:Z` on SELinux systems, e.g. `-v ./data:/data:Z`.

## Getting a YouTube API key

1. Create a project in the [Google Cloud Console](https://console.cloud.google.com/projectcreate).
2. Enable the [YouTube Data API v3](https://console.cloud.google.com/apis/library/youtube.googleapis.com).
3. [Credentials](https://console.cloud.google.com/apis/credentials) → **Create
   credentials → API key**. You can restrict the key to the YouTube Data API v3.
4. On YouTube, open [Settings → Privacy](https://www.youtube.com/account_privacy) and
   turn off **"Keep all my subscriptions private"**. A plain API key can only read
   public subscription lists.

For the channel, enter your `@handle`, your `UC…` channel id, or your channel URL.

If you'd rather keep your subscriptions private, skip the API key. Use **Setup → Import
from Google Takeout** instead. Without a key, DailyBee works the same but doesn't show
video lengths or view counts.

## Configuration

Everything can be set on the Setup page. These environment variables override it:

| Variable | Meaning |
| --- | --- |
| `DAILYBEE_YOUTUBE_API_KEY` | YouTube Data API v3 key |
| `DAILYBEE_YOUTUBE_CHANNEL` | your channel: `@handle`, `UC…` id or URL |
| `DAILYBEE_AUTH` | `username:password` to require a login |
| `DAILYBEE_ADDR`, `DAILYBEE_DB`, `DAILYBEE_BASE`, … | server options inherited from yarr (`dailybee -h` lists them all); the upstream `YARR_*` names still work |
| `TZ` | time zone, used in the container logs (the bulletin shows times in your browser's time zone) |

The API key is stored in the local database and is never sent to the browser.
The Setup page only shows its last four characters.

The binary refreshes channel feeds every 60 minutes by default. You can change this in
the reader's menu under **Auto refresh**.

## Pages and API

| Path | What |
| --- | --- |
| `/` | the bulletin (`?hours=48`, `?shorts=hide`) |
| `/setup` | API key, channel, options, Takeout import |
| `/reader` | the original yarr reader UI |
| `GET /api/bulletin?hours=24` | the bulletin as JSON |
| `GET/POST /api/youtube/sync` | subscription sync status / run a sync now |

All of yarr's own API endpoints (including the Fever API) still work.

## Development

```sh
npm ci && npm run build            # reader UI bundle (the bulletin pages need no build)
go run -tags "sqlite_foreign_keys sqlite_json sqlite_fts5" ./cmd/yarr -db dev.db
go test -short -tags "sqlite_foreign_keys sqlite_json sqlite_fts5" ./...
```

The DailyBee code is in `src/dailybee` (YouTube client, subscription sync, bulletin
building), `src/server/dailybee.go` (routes), `src/assets/templates/{bulletin,setup}.html`
and `src/assets/static/dailybee.{css,js}`. The rest is upstream yarr, with small hooks
added. That keeps it easy to merge upstream updates.

## License

MIT, like upstream. See [license](license). Based on yarr © Nazar Kanaev
(forked from [`nkanaev/yarr@aa29c4f`](https://github.com/nkanaev/yarr/commit/aa29c4f7bad6efa3f9cf92bfd5c3bb50325975fb)).
