-- Optional hardening for deployment: a role that can only read the v_reviews view.
-- Point AGENT_DATABASE_URL at this role so the agent cannot touch anything else, even if the
-- SQL guard in complaint_triage/sqlguard.py were bypassed. Run once as the database owner:
--   psql "$DATABASE_URL" -v pw="'choose-a-password'" -f sql/readonly_role.sql

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agent_ro') THEN
        CREATE ROLE agent_ro LOGIN;
    END IF;
END
$$;
ALTER ROLE agent_ro PASSWORD :pw;
ALTER ROLE agent_ro SET statement_timeout = '5s';
ALTER ROLE agent_ro SET default_transaction_read_only = on;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM agent_ro;
GRANT USAGE ON SCHEMA public TO agent_ro;
GRANT SELECT ON v_reviews TO agent_ro;
