# Makefile for RFT v2 Galaxy Rotation Curves Paper
#
# Targets:
#   make paper     - Build main.pdf from LaTeX sources
#   make figures   - Regenerate all figures from frozen JSON
#   make tables    - Regenerate all tables from frozen JSON
#   make all       - Build everything (figures, tables, paper)
#   make clean     - Remove build artifacts
#   make arxiv     - Prepare arXiv submission tarball
#   make verify    - Run reproducibility checks

.PHONY: all paper figures tables clean arxiv verify

# Main target: build paper
all: figures tables paper

# Build PDF from LaTeX
paper: main.pdf

build/numbers.tex: build/final_numbers.json
	@echo "===== Emitting LaTeX macros ====="
	cd .. && python3 scripts/emit_tex_numbers.py

main.pdf: main.tex sections/*.tex tables/*.tex build/numbers.tex
	@echo "===== Building paper ====="
	pdflatex main.tex
	bibtex main || true
	pdflatex main.tex
	pdflatex main.tex
	@echo "✅ Paper built: main.pdf"

# Regenerate all figures
figures:
	@echo "===== Generating figures ====="
	python3 figs/generate_f1_overview.py
	python3 figs/generate_f2_mcnemar.py
	python3 figs/generate_f3_lsb_hsb.py
	python3 figs/generate_f4_stability.py
	python3 figs/generate_f5_rotation_gallery.py
	python3 figs/generate_f6_ablations.py
	python3 figs/generate_f8_mechanism.py
	@echo "✅ All figures generated"

# Regenerate all tables
tables:
	@echo "===== Generating tables ====="
	python3 tables/generate_all_tables.py
	@echo "✅ All tables generated"

# Clean build artifacts
clean:
	@echo "===== Cleaning build artifacts ====="
	rm -f main.pdf main.aux main.log main.bbl main.blg main.out
	rm -f sections/*.aux
	@echo "✅ Clean complete"

# Prepare arXiv submission tarball
arxiv: figures tables
	@echo "===== Preparing arXiv submission ====="
	mkdir -p dist/arxiv
	cp main.tex dist/arxiv/
	cp -r sections dist/arxiv/
	cp -r figs/*.pdf dist/arxiv/figs/
	cp -r tables/*.tex dist/arxiv/tables/
	cp refs.bib dist/arxiv/
	cd dist/arxiv && tar czf ../rft-v2-arxiv.tar.gz *
	@echo "✅ arXiv tarball: dist/rft-v2-arxiv.tar.gz"

# Run reproducibility checks
verify:
	@echo "===== Running reproducibility checks ====="
	cd .. && python3 scripts/audit_baselines.py
	cd .. && python3 scripts/verify_final_numbers_hash.py
	cd .. && python3 scripts/verify_numbers.py
	@echo "✅ All gates passed"

# Help
help:
	@echo "RFT v2 Paper Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make all       - Build figures, tables, and paper (default)"
	@echo "  make paper     - Build main.pdf only"
	@echo "  make figures   - Regenerate all figures from frozen JSON"
	@echo "  make tables    - Regenerate all tables from frozen JSON"
	@echo "  make verify    - Run reproducibility checks (Gate 0, P1)"
	@echo "  make arxiv     - Prepare arXiv submission tarball"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make help      - Show this message"
