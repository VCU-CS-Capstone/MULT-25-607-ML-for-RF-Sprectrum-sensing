from pydantic import BaseSettings

class Settings(BaseSettings):
    sdr_sample_rate: float = 2.4e6  # Hz
    sdr_center_freq: float = 2.45e9  # Hz
    sdr_freq_correction: int = 60    # PPM
    sdr_gain: str = 'auto'
    model_path: str = "models/model.pth"
    websocket_endpoint: str = "/ws"
    allowed_origins: list = ["http://localhost:3000"]  # Frontend URL
    use_mock_sdr: bool = False
    use_file_sdr: bool = False
    use_psd_simulator: bool = False  # New flag for PSD simulator
    file_sdr_directory: str = "app/sdr/sample_data"
    psd_simulator_directory: str = "app/sdr/sample_data"  # Directory for PSD samples

    class Config:
        env_file = ".env"

settings = Settings()