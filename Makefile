
CU_VERSION = cu124

requirements:
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$(CU_VERSION)
	pip install -r requirements.txt