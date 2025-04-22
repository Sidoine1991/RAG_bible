export GCP_SA_KEY_JSON="$(cat path/to/sa-key.json)"
export GCP_PROJECT_ID="biblioIA"
export GCP_REGION="europe-west1"
export GOOGLE_API_KEY="AIzaSyA1uSrTi2wqDnpYV4FRqkPDKFEah8cIFuM"

gh repo set-secret GCP_SA_KEY --body "$GCP_SA_KEY_JSON"
gh repo set-secret GCP_PROJECT_ID --body "$GCP_PROJECT_ID"
gh repo set-secret GCP_REGION --body "$GCP_REGION"
gh repo set-secret GOOGLE_API_KEY --body "$GOOGLE_API_KEY"