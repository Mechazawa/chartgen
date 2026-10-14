# Charting from anywhere: Cloudflare Tunnel + Access

The web UI (`scripts/chartgen-web.sh`) turns the PC with the GPU into a private
charting service for you and people you choose. Nothing is exposed on your
router, your home IP is never published, and a login wall sits in front of
the app before any request reaches your machine. All of it is on
Cloudflare's free tier.

How it behaves once it is up:

- Everyone signs in through Cloudflare Access (email one-time PIN, or a
  Google/GitHub login) and only sees their own queue.
- Finished songs download to the visitor's device automatically.
- **Owners** (the emails you list in `CHARTGEN_WEB_OWNERS`) get their songs
  saved to the PC's Clone Hero library as well, exactly like the desktop app.
- **Guests** (anyone else on the allowlist) get their song charted in a
  scratch folder, handed over as a zip, and deleted from the PC once they
  have downloaded it. Their uploaded audio is deleted as soon as charting
  finishes. Nothing of theirs stays unless they tick "save to this PC's
  library".
- Anything a guest never downloads is swept after 24 hours.

The PC has to be on for any of this to work; if it sleeps, the tunnel drops.

## 1. Install the tunnel (Cloudflare dashboard + one admin command)

1. Cloudflare dashboard → **Zero Trust** → **Networks** → **Tunnels** →
   **Create a tunnel** → *Cloudflared* → name it `chartgen`.
2. Pick your Linux distribution. The dashboard shows how to install
   `cloudflared` and a `sudo cloudflared service install <token>` command.
   Run both. That installs `cloudflared` as a systemd service that starts
   with the PC.
3. Under **Public Hostname** → **Add a public hostname**:
   subdomain `app`, domain `chartgen.org`, type **HTTP**,
   URL `127.0.0.1:8471`. Save. Cloudflare creates the DNS record itself.

## 2. Put a login in front of it

1. **Zero Trust** → **Access** → **Applications** → **Add an application** →
   *Self-hosted*. Name `chartgen`, domain `app.chartgen.org`.
2. Add a policy: name `allowed`, action **Allow**, include **Emails** and list
   every address that may use it (yours first). Save.
3. Under **Settings** → **Authentication**, make sure **One-time PIN** is
   enabled (it is by default). Visitors get a code by email; no accounts.

Anyone not on the list hits Cloudflare's login page and never reaches your
PC. The app trusts the `Cf-Access-Authenticated-User-Email` header Access
adds, which is why `chartgen-web` listens on `127.0.0.1` only: nothing on
your network can reach it without going through Access.

## 3. Tell chartgen who the owners are

The owners are the emails in `CHARTGEN_WEB_OWNERS`. Comma-separate several
addresses. Anyone signed in with another address is a guest. For a quick
test, set it on the command line:

```
CHARTGEN_WEB_OWNERS=you@example.com scripts/chartgen-web.sh
```

## 4. Keep the web UI running

Run it as a systemd user service. Create
`~/.config/systemd/user/chartgen-web.service`:

```
[Unit]
Description=chartgen web UI

[Service]
ExecStart=/path/to/chartgen/scripts/chartgen-web.sh
Environment=CHARTGEN_WEB_OWNERS=you@example.com
Restart=on-failure

[Install]
WantedBy=default.target
```

Then start it, and let it run without a login session:

```
systemctl --user daemon-reload
systemctl --user enable --now chartgen-web
loginctl enable-linger "$USER"
```

Read its log with `journalctl --user -u chartgen-web -f`.

## 5. Test it

From a phone on mobile data: open `https://app.chartgen.org`, sign in with
the emailed code, paste a link. About 90 seconds later the zip arrives.

## Stopping

Disable the Access application (or remove an email from the policy) to cut
someone off instantly; delete the tunnel to take the whole thing down.

## What this is not

A private tool for you and people you know, which is what keeps it on the
right side of the copyright line — the same posture as running chartgen
locally, with a longer cable. It is not a public "paste a link" service:
never widen the Access policy to "everyone".
