from pydantic_settings import BaseSettings,SettingsConfigDict


class Settings(BaseSettings):   
    OPENAI_API_KEY : str
    INDEX_NAME :str
    
    model_config = SettingsConfigDict(env_file=".env")
    
Setting = Settings()