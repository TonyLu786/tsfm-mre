TSFM-MRE src/ quickstart
========================
1) Ingest data (synthetic by default):
   python src/data_ingest.py --outdir data/raw --mode synthetic --start 2003-01-01 --end 2025-06-30

2) Preprocess & split:
   python src/preprocess.py --indir data/raw --outdir data/interim --train_start 2003-01-01 --valid_start 2013-01-02 --test_start 2019-01-02 --end 2025-06-30

3) Forecast (AR(1) baseline; tiny_tf optional if PyTorch installed):
   python src/forecast.py --indir data/interim --outdir outputs/forecasts --model ar1 --horizons 1 5 10 20 --context 60

4) Risk backtesting (99% VaR; h=1):
   python src/risk.py --indir data/interim --pred outputs/forecasts/pred_ar1.parquet --outdir outputs/risk --alpha 0.01 --horizon 1 --window 250

5) Plots:
   python src/plots.py --indir data/interim --pred outputs/forecasts/pred_ar1.parquet --risk_dir outputs/risk --outdir outputs/figures --alpha 0.01 --horizon 1

6) Report:
   python src/report.py --risk_dir outputs/risk --fig_dir outputs/figures --outdir outputs/report

All scripts run offline with the synthetic data path. For FRED downloads, use --mode fred in data_ingest.py if pandas_datareader is installed.