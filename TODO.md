# ytauto — TODO (baad ke liye)

## Network / Hosting
- [ ] **Netlify hosting (dashboard ka static snapshot)** — har hourly cycle ke end mein static site build karke Netlify pe auto-deploy (API + data embedded, max 1hr fresh). User ko ek baar Netlify login karwana hoga (netlify CLI). User ne bola "baad me karenge" (2026-10-18).
- [ ] (Isi se juda) Live-tunnel option — Mac ka port expose karna padta, risky; na chahiye to skip.

## Ideas (user ne kabhi bola ho to)
- [ ] Rumble category auto-select (Education/News) — custom dropdown widget, abhi skip ho raha hai warning ke saath.
- [ ] **YouTube publishing ON karna** — सब ready baitha hai: OAuth creds `secrets/` mein, upload-scope authorized, channel linked (UCbFpK2bw1Px4WL6T9AYxBHw), run-loop ka stage-7 upload_queue har cycle dry-run mein chalta hai. Sirf `.env` mein `DRY_RUN=0` karna hoga — next cycle se har video Rumble + YouTube dono pe. Yaad rakhna: Google OAuth testing mode = ~7 din mein `ytauto auth` se re-login; new-channel upload quota ~6/day. User ne bola "abhi rehne do" (2026-10-18; base language English rakhi gayi US audience ke liye, sidecars Hindi/Bengali/French).
