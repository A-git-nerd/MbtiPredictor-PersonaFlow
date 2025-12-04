import re
import os

def clean_chat(path):
    f = open(path, "r", encoding="utf-8")
    txt = f.read().splitlines()
    f.close()

    p = r"^\d{1,2}\/\d{1,2}\/\d{2,4},\s\d{1,2}:\d{1,2}\s?(am|pm)?\s-\s"
    cur = ""
    out = []

    for line in txt:
        if re.match(p, line):
            if cur.strip() != "":
                if "<Media omitted>" not in cur and "This message was deleted" not in cur:
                    out.append(cur.replace("\n"," ").strip())
            cur = line
        else:
            cur += " " + line.strip()

    if cur.strip() != "":
        if "<Media omitted>" not in cur and "This message was deleted" not in cur:
            out.append(cur.strip())

    t = "temp.txt"
    w = open(t, "w", encoding="utf-8")
    for m in out:
        w.write(m + "\n")
    w.close()

    os.replace(t, path)
