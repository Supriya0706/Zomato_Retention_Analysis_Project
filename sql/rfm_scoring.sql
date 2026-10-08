-- RFM (Recency, Frequency, Monetary) scoring using NTILE window functions.
-- Each dimension is split into quartiles (1 = worst, 4 = best), then
-- combined into a single rfm_segment label. This is the standard,
-- industry-recognized approach to customer segmentation.
--
-- Recency  : last_order_days   (LOWER is better -> invert the quartile)
-- Frequency: orders            (HIGHER is better)
-- Monetary : total_spend       (HIGHER is better)

WITH rfm_base AS (
    SELECT
        user_id,
        last_order_days,
        orders,
        total_spend,
        -- Recency: lower last_order_days = better, so quartile 4 = most recent
        5 - NTILE(4) OVER (ORDER BY last_order_days ASC) AS recency_score,
        NTILE(4) OVER (ORDER BY orders ASC)               AS frequency_score,
        NTILE(4) OVER (ORDER BY total_spend ASC)          AS monetary_score
    FROM dim_users
),
rfm_scored AS (
    SELECT
        *,
        (recency_score + frequency_score + monetary_score) AS rfm_total,
        CASE
            WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3
                THEN 'Champions'
            WHEN recency_score >= 3 AND frequency_score <= 2
                THEN 'Recent but Low Frequency'
            WHEN recency_score <= 2 AND frequency_score >= 3 AND monetary_score >= 3
                THEN 'At-Risk High Value'
            WHEN recency_score <= 2 AND frequency_score <= 2
                THEN 'Dormant / Lost'
            ELSE 'Regular'
        END AS rfm_segment
    FROM rfm_base
)
SELECT
    rfm_segment,
    COUNT(*)                         AS user_count,
    ROUND(AVG(orders), 1)            AS avg_orders,
    ROUND(AVG(total_spend), 2)       AS avg_total_spend,
    ROUND(AVG(last_order_days), 1)   AS avg_days_since_order
FROM rfm_scored
GROUP BY rfm_segment
ORDER BY user_count DESC;
