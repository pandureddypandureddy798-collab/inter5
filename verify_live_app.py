import subprocess
import time
import urllib.request
import sys

print("Starting Streamlit in headless mode on port 8501...")
proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

try:
    # Wait up to 15 seconds for server to come up
    success = False
    for i in range(15):
        time.sleep(1)
        try:
            resp = urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=2)
            if resp.status == 200:
                print("Streamlit health check passed with HTTP 200!")
                success = True
                break
        except Exception:
            pass

    if not success:
        print("Could not reach Streamlit health endpoint within 15 seconds.")
        sys.exit(1)
    else:
        # Also check root endpoint
        root_resp = urllib.request.urlopen("http://localhost:8501/", timeout=2)
        print(f"Streamlit root endpoint HTTP status: {root_resp.status}")
        assert root_resp.status == 200
        print("Streamlit app is working and responding perfectly!")

finally:
    print("Terminating test Streamlit server process...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("Streamlit process stopped.")
