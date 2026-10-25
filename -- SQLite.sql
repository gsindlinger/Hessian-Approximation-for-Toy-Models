  SELECT
      reference,
      computation_type,
      COUNT(*) AS rows,
      COUNT(DISTINCT approximator) AS approximators,
      COUNT(DISTINCT metric) AS metrics
  FROM metrics
  WHERE result_id = 236
  GROUP BY reference, computation_type
  ORDER BY reference, computation_type;