import os
import traceback

from copilot import SalesCopilot

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "googleServiceAccountCredentials.json"
os.environ["PINECONE_API_KEY"] = "pcsk_3DxeCk_EBLSPYzsFox6yaa6ngWSQ7QLZGkna9nt45WBzsVfZq6wKdhm3uPRNBA4xpjthLT"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAnJg3id4YD6DT5wFAittSH_BGHv6mALvU"
credentials_path = "googleServiceAccountCredentials.json"


def main():
    """Main entry point for the application."""
    # Create and run the copilot
    try:
        copilot = SalesCopilot(credentials_path=credentials_path)

        # Run the copilot
        copilot.run()

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
    finally:
        print("Copilot terminated.")


if __name__ == "__main__":
    main()
