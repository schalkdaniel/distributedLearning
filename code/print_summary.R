# Get the environment variables
model_dir <- Sys.getenv("MODEL_DIR")
code_dir <- Sys.getenv("CODE_DIR")
data_dir <- Sys.getenv("DATA_DIR")

# Change to the distributed learning dir and load stuff
setwd(code_dir)
devtools::load_all()
