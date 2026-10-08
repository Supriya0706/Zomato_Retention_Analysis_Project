-- Cohort retention analysis: for each signup month, what % of that cohort
-- is still active N months later? This is the classic cohort heatmap
-- structure (rows = signup month, columns = months since signup).
--
-- Uses a CTE to first tag each user with their signup (cohort) month, then
-- joins against monthly activity and aggregates retention % per cohort per
-- month-since-join.

WITH cohorts AS (
    SELECT
        user_id,
        strftime('%Y-%m', join_date) AS cohort_month
    FROM dim_users
),
cohort_activity AS (
    SELECT
        c.cohort_month,
        a.months_since_join,
        a.user_id,
        a.is_active
    FROM fact_activity a
    JOIN cohorts c ON c.user_id = a.user_id
)
SELECT
    cohort_month,
    months_since_join,
    COUNT(DISTINCT user_id)                           AS cohort_users_observed,
    SUM(is_active)                                     AS active_users,
    ROUND(100.0 * SUM(is_active) / COUNT(DISTINCT user_id), 1) AS retention_pct
FROM cohort_activity
GROUP BY cohort_month, months_since_join
ORDER BY cohort_month, months_since_join;
