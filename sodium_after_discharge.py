"""Count outcome-window sodiums recorded after ICU discharge."""
import os
import psycopg2

os.environ.setdefault("PGPASSWORD", "local")


def q(db, sql):
    conn = psycopg2.connect(host="127.0.0.1", port=5442, user="postgres", dbname=db, password=os.environ["PGPASSWORD"])
    cur = conn.cursor()
    cur.execute(sql)
    row = cur.fetchone()
    conn.close()
    return row


print("mimic", q("mimiciv31", """
WITH cohort AS (
  SELECT f.stay_id, i.intime, i.outtime
  FROM screen.paper_mimic p
  JOIN screen.features f ON f.stay_id = p.stay_id
  JOIN mimiciv_icu.icustays i ON i.stay_id = p.stay_id
),
na AS (
  SELECT a.stay_id,
         c.charttime >= a.outtime AS after_out
  FROM cohort a
  JOIN screen.adult_first af ON af.stay_id = a.stay_id
  JOIN mimiciv_derived.chemistry c
    ON c.hadm_id = af.hadm_id
   AND c.charttime >= a.intime
   AND c.charttime < a.intime + interval '72 hours'
   AND c.charttime >= a.intime + interval '24 hours'
   AND c.sodium BETWEEN 100 AND 180
)
SELECT
  count(*) AS sodiums,
  count(*) FILTER (WHERE after_out) AS after_discharge,
  count(DISTINCT stay_id) AS stays,
  count(DISTINCT stay_id) FILTER (WHERE after_out) AS stays_with_after
FROM na
"""))

print("eicu", q("eicu", """
WITH na AS (
  SELECT p.stay_id,
         l.labresultoffset >= pt.unitdischargeoffset AS after_out
  FROM screen.paper_eicu p
  JOIN patient pt ON pt.patientunitstayid = p.stay_id
  JOIN lab l ON l.patientunitstayid = p.stay_id
  WHERE lower(l.labname) = 'sodium'
    AND l.labresultoffset >= 24*60
    AND l.labresultoffset < 72*60
    AND l.labresult BETWEEN 100 AND 180
)
SELECT
  count(*) AS sodiums,
  count(*) FILTER (WHERE after_out) AS after_discharge,
  count(DISTINCT stay_id) AS stays,
  count(DISTINCT stay_id) FILTER (WHERE after_out) AS stays_with_after
FROM na
"""))

print("left before 24h mimic", q("mimiciv31", """
SELECT count(*)
FROM screen.paper_mimic p
JOIN mimiciv_icu.icustays i ON i.stay_id = p.stay_id
WHERE i.outtime <= i.intime + interval '24 hours'
"""))
print("left before 24h eicu", q("eicu", """
SELECT count(*)
FROM screen.paper_eicu p
JOIN patient pt ON pt.patientunitstayid = p.stay_id
WHERE pt.unitdischargeoffset <= 24*60
"""))
