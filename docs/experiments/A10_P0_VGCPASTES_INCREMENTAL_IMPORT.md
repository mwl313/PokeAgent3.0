# A10-P0 VGCPastes incremental import

Capture date: 2026-10-12 (UTC). Authoritative source is the Google Sheet `1axlwmzPA49rYkqXh7zHvAtSP-TKbM0ijGYBPRflLSWw`, tabs `Champions M-C` (gid 2001945654) and `Champions M-B` (gid 1458357160). The prior boundary is MC463/MB861 from the 2026-10-06 snapshots. The source CSVs, live sheet snapshots, fetched HTML, extracted text, metadata, and hashes are retained under `data/raw/vgcpastes/`.

M-C advanced to MC584. There were 121 new rows, 97 EVs=Yes rows, and 97 fetched Pokepaste pages. Pinned Showdown validation accepted 85 records and quarantined 12. Two duplicate groups occurred within the new candidates (MC469/MC533 and MC515/MC568), leaving 83 unique additions. No new candidate duplicates a v3 team. M-B remains MB861, with zero new rows or candidates.

The comparison also detected one metadata-only edit to existing MC189 (`report_url`); its team was not reimported. No existing Pokepaste link changed. The live sheets contain pre-existing duplicate-link groups MC533/MC497 and MB843/MB791; these were recorded and do not increase the inventory.

The proposed immutable successor is `data/teams/a10-p0-proposed-20261013`, with 1,220 unique eligible teams (1,137 predecessor teams verified byte-identical plus 83 additions). All accepted rows remain in the training split by the inherited all-train policy; this is a proposed pool, not a production switch. Quarantined rows are preserved with reasons and are not silently repaired. The raw source is public provenance; redistribution permission is not inferred.

The import is complete for P0. Native engine pool readiness is reported separately and remains NOT VERIFIED until the authoritative Rust toolchain can run the pool-scoped checks and bounded whole-pool natural battles. No PPO training or optimizer update was started.
