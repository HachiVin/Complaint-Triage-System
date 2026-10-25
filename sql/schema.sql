-- Complaint Triage System schema (PostgreSQL 14+). Applied by the API and by scripts/ingest.py; safe to re-run.

CREATE TABLE IF NOT EXISTS reviews (
    id           BIGSERIAL PRIMARY KEY,
    source       TEXT        NOT NULL,                 -- ush_brimo, play_scrape, synthetic, api
    external_id  TEXT        NOT NULL,                 -- id in the source; reviewer names are never stored
    app          TEXT        NOT NULL,
    text         TEXT        NOT NULL,
    rating       SMALLINT    CHECK (rating BETWEEN 1 AND 5),
    review_date  DATE,
    app_version  TEXT,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, external_id)
);
CREATE INDEX IF NOT EXISTS reviews_app_date_idx ON reviews (app, review_date);

-- Ground-truth labels: from the dataset, from annotation, or from a CX agent correcting the model.
CREATE TABLE IF NOT EXISTS labels (
    review_id   BIGINT PRIMARY KEY REFERENCES reviews (id) ON DELETE CASCADE,
    aspect      TEXT,
    sentiment   TEXT,
    urgency     TEXT,
    labelled_by TEXT NOT NULL,                          -- dataset, annotator, cx
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS predictions (
    id                 BIGSERIAL PRIMARY KEY,
    review_id          BIGINT      NOT NULL REFERENCES reviews (id) ON DELETE CASCADE,
    model_version      TEXT        NOT NULL,
    aspect             TEXT        NOT NULL,
    aspect_conf        REAL        NOT NULL,
    sentiment          TEXT        NOT NULL,
    sentiment_conf     REAL        NOT NULL,
    urgency            TEXT        NOT NULL,
    urgency_conf       REAL        NOT NULL,
    urgency_source     TEXT        NOT NULL,            -- model or rules
    needs_review       BOOLEAN     NOT NULL,
    predicted_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (review_id, model_version)
);
CREATE INDEX IF NOT EXISTS predictions_review_idx ON predictions (review_id, predicted_at DESC);

CREATE TABLE IF NOT EXISTS corrections (
    id          BIGSERIAL PRIMARY KEY,
    review_id   BIGINT      NOT NULL REFERENCES reviews (id) ON DELETE CASCADE,
    task        TEXT        NOT NULL CHECK (task IN ('aspect', 'sentiment', 'urgency')),
    predicted   TEXT,
    corrected   TEXT        NOT NULL,
    model_version TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_log (
    id                BIGSERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    question          TEXT        NOT NULL,
    queries           JSONB       NOT NULL DEFAULT '[]',
    answer            TEXT,
    ok                BOOLEAN     NOT NULL,
    error             TEXT,
    steps             INT,
    latency_ms        REAL,
    prompt_tokens     INT,
    completion_tokens INT,
    llm_model         TEXT
);

-- The one view the agent and the dashboard read: each review with its latest prediction.
CREATE OR REPLACE VIEW v_reviews AS
SELECT r.id            AS review_id,
       r.app,
       r.text,
       r.rating,
       r.review_date,
       r.app_version,
       p.aspect,
       p.sentiment,
       p.urgency,
       p.aspect_conf,
       p.sentiment_conf,
       p.urgency_conf,
       p.needs_review,
       p.model_version
FROM reviews r
JOIN LATERAL (
    SELECT * FROM predictions p WHERE p.review_id = r.id ORDER BY p.predicted_at DESC LIMIT 1
) p ON true;
