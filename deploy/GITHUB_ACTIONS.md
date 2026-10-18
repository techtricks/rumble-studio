# GitHub Actions par automation (Mac band => automation zinda)

Ye guide Mac ke **band hone per bhi** ek video per day ka automation chalata hai
— GitHub ke free runners pe, **bina kisi credit card**.
Base voices Edge-TTS neural hain (Mac ke nahi), publish source Rumble hai.

## Aapko kya karani hai (5 min)

1. **Repo bana do** (public hona zaroori hai — unlimited free minutes)
   - github.com pe New repository -> nam e.g. `rumble-studio` -> Create

2. **Secrets daalo** (repo -> Settings -> Secrets and variables -> Actions -> New repository secret)

   | Secret | Value |
   |---|---|
   | `GROQ_API_KEY` | `.env` me jo hai (shuru mein `gsk_...`) |
   | `RUMBLE_EMAIL` | aapki Rumble login email |
   | `RUMBLE_PASSWORD` | Rumble password |
   | `RUMBLE_CHANNEL` | e.g. `IndianHistoryOfficial` |
   | `RUMBLE_COOKIES_B64` | Neeche wala command niche paste kiya hua output |

   Cookies ka secret (Mac pe ek baar):
   ```bash
   base64 -i "/Users/mac/Documents/rumble/secrets/rumble_cookies.json" | pbcopy
   # ab clipboard ka content paste karo RUMBLE_COOKIES_B64 field mein
   ```
   Cookies expire ho to (upload warn dekha) to wahi command phir se chalayein.

3. **Push karo:**
   ```bash
   cd "/Users/mac/Documents/rumble"
   git init -b main
   git add -A && git commit -m "pipeline: always-on CI"
   git remote add origin https://github.com/<YOUR-USERNAME>/rumble-studio.git
   git push -u origin main
   ```

4. **Pehli baar rehearsal:** Actions -> ytauto-cycle -> Run workflow ->
   dabake `dry_run = true` check karke chla do. पूरा cycle dry-run mode mein
   chalega (kuch publish nahi hoga) aur batayega kya galat hai.

5. **Live:** dry_run ke bagair chladne par roj 20:00 IST pe chal jaayega:
   longform + uska short dono Rumble pe. State beech me `ytauto-state` artifact
   me travel karta hai (topic-pool, published status, used-marker sab safe).

## Zaroori jaanken

- **Cloudflare:** runner ki IP kabhi kabhi Rumble ke challenge pe atak ji
  sakti hai. Step fail hota hai to us- din upload nahi hui; DB artifact maaf
  hai aur agli raat dobara koshish. Mac fir khali hai to Mac pe `ytauto rumble`
  chalake manual push kare.
- **Mac pe jo abhi wala LaunchAgent (hourly cycle) chal raha hai**: CI ON ke
  baad usse rukna chahiye, warna +2 videos/day (duplicate bhi chal sakega):
  ```bash
  launchctl bootout gui/$UID/com.ytauto.cycle
  ```
  (Dashboard ka ek separate agent hai, wahan change karne ki koi zarurat nahi.)
- **YouTube abhi rehearsal hai** (`DRY_RUN=1`), jaisay humne decide kiya tha.
- Quota: ek public repo pe monthly minutes unlimited, aur roz ONE cycle minna
  1 video/day, bohot pairavi hai.
