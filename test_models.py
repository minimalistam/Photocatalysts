
from inference_utils import load_model_pipeline

def test_model_loading():
    print("Testing Model Loading...")
    
    # Test Perovskite
    try:
        print("\nLoading Perovskite Model...")
        model_p, manifest_p, encoders_p = load_model_pipeline("models/perovskite")
        print("✅ Perovskite Model loaded successfully")
        print(f"   Version: {manifest_p.get('pipeline_version', 'Unknown')}")
    except Exception as e:
        print(f"❌ Failed to load Perovskite model: {e}")
        
    # Test Spinel
    try:
        print("\nLoading Spinel Model...")
        model_s, manifest_s, encoders_s = load_model_pipeline("models/spinel")
        print("✅ Spinel Model loaded successfully")
        print(f"   Version: {manifest_s.get('pipeline_version', 'Unknown')}")
    except Exception as e:
        print(f"❌ Failed to load Spinel model: {e}")

if __name__ == "__main__":
    test_model_loading()
