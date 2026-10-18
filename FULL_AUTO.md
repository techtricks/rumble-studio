# ytauto full-auto — 1-page cheat sheet (chat kholne ki zaroorat nahi)

System har ghanta APNE AAP chalata hai. Aap sirf yeh check kar sakte ho:

## 🔴 LIVE DASHBOARD (sabse pehle yeh dekho!)
```bash
open http://localhost:8600
```
- **Login chahiye hi nahi** — browser mein khulte hi sab dikhega
- Live status (khulta hai hi nahi): kaunsa cycle chal raha, videos kahan atki, topics queue, published count, poora log
- Har 5 second auto-refresh
- Mac login pe apne aap start hota hai (LaunchAgent) — kuch karne ki zaroorat nahi
- Dashboard band ho jaye toh: `launchctl load ~/Library/LaunchAgents/com.ytauto.dashboard.plist`

## Ek nazar mein sab (status + queue)
```bash
cd /Users/mac/Documents/rumble
.venv/bin/python -m ytauto status
```
- Channel info, videos in queue, aur aaj ka schedule dikhata hai

## Scheduler ka log (ghanton ke cycles)
```bash
tail -40 data/logs/launchd.log
```
`cycle start/end` + stage summaries. Errors dhundte toh `grep XX data/logs/launchd.log`

## Published videos dekho (Rumble)
- Channel: https://rumble.com/c/IndianHistoryOfficial
- Dashboard: https://rumble.com/account/content

## Scheduler control
```bash
launchctl list | grep ytauto          # chal raha hai? (- = idle, exit 0 = thik)
launchctl list com.ytauto.rumble      # details
launchctl unload ~/Library/LaunchAgents/com.ytauto.rumble.plist   # BAND
launchctl load   ~/Library/LaunchAgents/com.ytauto.rumble.plist   # CHALU
```

## Mac sleep (yeh aapko EK baar karna hai, sudo ke saath)
```bash
sudo pmset -a sleep 0              # system sleep OFF
sudo pmset -a displaysleep 10      # screen sleep theek hai
```
(NEVER close the lid on MacBook — clamshell = system sleep. Desktop Mac hai toh no tension)

## Agar koi cheez atak jaye
- `ytauto script --review` — kis stage pe kya ruka
- `ytauto render --force` — render retry
- `ytauto rumble` — Rumble upload manual push (cookies already saved hain)

## Safety rails jo ON hain
- AUTO_APPROVE=1 lekin: research-confidence, niche (Indian history), duplicate & originality gates ke bina script auto-publish NAHI hota
- 6cc/CC-license images only, rotating voices — monetisation-safe defaults
