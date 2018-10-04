# Get the environment variables
code_dir <- Sys.getenv("CODE_DIR")
model_dir <- Sys.getenv("MODEL_DIR")

# Change to the distributed learning dir and load stufg
setwd(code_dir)
devtools::load_all()

# Setup file path
model_file <- paste(model_dir, "train_files/model.rds", sep="/")
regis_dir <- paste(model_dir, "train_files", sep="/")

# Load the model
load(model_file)

# Checks whether the training has been completed
# The model will tell  us that
# Also, this function breaks the execution of this script (note that the exit code must be 0, otherwise the training
# has officially failed.
check_training_done <- function() {

	if(model[["done"]]) {

  		flag_finish_file = paste(model_dir, "FINISH", sep="/")

		# If the file already exists, we will need to delete it first
		if(file.exists(flag_finish_file)) {

			# The training fails if the removal of the file fails
			if( ! file.remove(flag_finish_file)) {
				warning("FLAG file for termination already exists")
				quit(save="no", status=1)	
			}
		}

		# Now recreate the file.
		# If the creation of the file fails, we also need to cancel the training 
  		if( ! file.create(flag_finish_file)) {
			warning("FLAG file for termination could not be created successfully!")
			quit(save="no", status=2)
		}

		# Otherwise, everything is fine and the process can be terminated normally
		quit(save = "no", status = 0)
	}
}

# We check whether the model is finished before and after we train it
check_training_done()
trainDistributedModel(regis = regis_dir, epochs_at_once = 10L)
check_training_done()

