# Data Provenance & Licensing Log

This is the project's legal/ethical record of where every training image came
from. It is the defense if licensing is ever questioned at exhibition, and it
backs the artist statement. **Prefer public-domain / CC sources.**

`scripts/01_scrape_data.py` appends one row per downloaded image automatically.
Add any **manually collected** images by hand using the same columns.

## Sources used (summary)

| Source | API / URL | License basis | Notes |
|---|---|---|---|
| The Met — Open Access | https://metmuseum.github.io/ | CC0 (public domain, `isPublicDomain=true` only) | No API key. We download only objects flagged public-domain. |
| Wikimedia Commons | https://commons.wikimedia.org/w/api.php | Per-file (PD / CC-BY / CC-BY-SA) — see `license` column | Bot policy requires a descriptive User-Agent; we send one with contact email. |

## Downloaded images

<!-- 01_scrape_data.py appends rows below this line. Do not delete the header. -->

| filename | class | source | original_url | license |
|---|---|---|---|---|
