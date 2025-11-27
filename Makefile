PYTHON_VERSION ?= 3.10

.PHONY: help
help: ## 显示帮助文件
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {sub("\\\\n",sprintf("\n%22c"," "), $$2);printf "\033[36m%-40s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: setup-hooks
setup-hooks: ## 配置 pre-commit hooks
	@echo "Setting up pre-commit hooks..."
	@if [ ! -d ".git" ]; then \
		echo "⚠️  Not a Git repository, skipping hooks installation"; \
	else \
		git config --local core.hooksPath "./scripts/githooks" ; \
		echo "✅ Git pre-commit hook installed (linked to scripts/pre-commit)"; \
	fi

.PHONY: fmt
fmt:
	@uv run isort agentrun
	@uv run isort tests
	@uv run isort examples
	@uv run find -L ./agentrun -not -path "*/.*" -type f -name "*.py" -exec pyink --config pyproject.toml {} +
	@uv run find -L ./tests -not -path "*/.*" -type f -name "*.py" -exec pyink --config pyproject.toml {} +
	@uv run find -L ./examples -not -path "*/.*" -type f -name "*.py" -exec pyink --config pyproject.toml {} +

# 只格式化指定的文件
.PHONY: fmt-file
fmt-file:
	@if [ -z "$(FMT_FILE)" ]; then \
		echo "Usage: make fmt-file FMT_FILE=path/to/file.py"; \
		exit 1; \
	fi
	@if [ -f "$(FMT_FILE)" ]; then \
		echo "Formatting $(FMT_FILE)"; \
		uv run isort "$(FMT_FILE)"; \
		uv run pyink --config pyproject.toml "$(FMT_FILE)"; \
	else \
		echo "File $(FMT_FILE) does not exist"; \
		exit 1; \
	fi


JINJA2_FILES := \
	agentrun/agent_runtime/api/control.py \
	agentrun/credential/api/control.py \
	agentrun/model/api/control.py \
	agentrun/toolset/api/control.py \
	agentrun/sandbox/api/control.py
JINJA2_CONFIGS := \
	codegen/configs/agent_runtime_control_api.yaml \
	codegen/configs/credential_control_api.yaml \
	codegen/configs/model_control_api.yaml \
	codegen/configs/toolset_control_api.yaml \
	codegen/configs/sandbox_control_api.yaml \

define make_jinja2_rule
$(1): $(2)
	@echo "Generating $$@ from $$<"
	@uv run python3 codegen/codegen.py --jinja2-only --config "$$<"
endef

# 应用 Jinja2 规则
$(eval $(call make_jinja2_rule,agentrun/agent_runtime/api/control.py,codegen/configs/agent_runtime_control_api.yaml))
$(eval $(call make_jinja2_rule,agentrun/credential/api/control.py,codegen/configs/credential_control_api.yaml))
$(eval $(call make_jinja2_rule,agentrun/model/api/control.py,codegen/configs/model_control_api.yaml))
$(eval $(call make_jinja2_rule,agentrun/toolset/api/control.py,codegen/configs/toolset_control_api.yaml))
$(eval $(call make_jinja2_rule,agentrun/sandbox/api/control.py,codegen/configs/sandbox_control_api.yaml))
TEMPLATE_FILES := $(shell find . -name "__*async_template.py" -not -path "*__pycache__*" -not -path "*egg-info*")

# 根据模板文件生成对应的输出文件路径
define template_to_output
$(patsubst ./%,%,$(subst __,,$(subst _async_template.py,.py,$1)))
endef

# 生成所有输出文件的列表
SYNC_FILES := $(foreach template,$(TEMPLATE_FILES),$(call template_to_output,$(template)))

# 为每个模板文件创建依赖关系规则
define make_sync_rule
$(call template_to_output,$1): $1
	@echo "Generating $$@ from $$<"
	@uv run python3 codegen/codegen.py --sync-only --template "$$<"
	@make fmt-file FMT_FILE="$$<"
endef

# 应用规则到所有模板文件
$(foreach template,$(TEMPLATE_FILES),$(eval $(call make_sync_rule,$(template))))

.PHONY: codegen
codegen: $(JINJA2_FILES) $(SYNC_FILES) ## 生成代码

.PHONY: sync_codegen
sync_codegen:
	@uv run python3 codegen/codegen.py --sync-only
	@uv run codegen/codegen.py --sync-only

.PHONY: codegen_file
codegen_file:
	@uv run python3 codegen/codegen.py --jinja2-only

.PHONY: codegen-all
codegen-all: sync_codegen codegen_file fmt ## 强制重新生成所有代码


.PHONY: setup
setup: install-uv install-deps setup-hooks


.PHONY: install-uv
install-uv: ## Check and install uv
	@if command -v uv >/dev/null 2>&1; then \
		echo "✅ uv is already installed: $$(uv --version)"; \
	else \
		pip install uv; \
	fi

.PHONY: install-deps
install-deps:
	command -v uv >/dev/null 2>&1 && \
		uv sync \
		--python ${PYTHON_VERSION} \
		--dev \
		--all-extras \
		-i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
