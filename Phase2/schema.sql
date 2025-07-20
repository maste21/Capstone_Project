-- Step 1: Create final feature table
CREATE TABLE GreenEnergy_DBP.dbo.energy_features (
    datetime DATETIME,
    hour INT,
    day DATE,
    avg_consumption_kwh FLOAT,
    peak_consumption_kwh FLOAT,
    avg_temperature FLOAT,
    avg_humidity FLOAT,
    avg_wind_speed FLOAT,
    solar_output_kwh FLOAT
);
-- Step 2: Index on day for fast queries
CREATE NONCLUSTERED INDEX idx_energy_day ON GreenEnergy_DBP.dbo.energy_features(day);

-- Step 3 (Optional): Create view for Power BI
CREATE VIEW vw_daily_consumption_summary AS
SELECT 
    CAST(datetime AS DATE) AS day,
    AVG(avg_consumption_kwh) AS daily_avg_kwh,
    MAX(peak_consumption_kwh) AS peak_kwh,
    SUM(solar_output_kwh) AS total_solar
FROM GreenEnergy_DBP.dbo.energy_features
GROUP BY CAST(datetime AS DATE);
