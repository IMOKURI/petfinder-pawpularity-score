.PHONY: help
.DEFAULT_GOAL := train

NOW = $(shell date '+%Y%m%d-%H%M%S')

train: ## Run training
	@python train.py


debug_train: ## Run training with debug
	@python train.py settings.debug=True


help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[38;2;98;209;150m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
