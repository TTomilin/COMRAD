# An example of integrating wad2image into a Makefile.

clean: wad-image-clean
	@echo "Whatever other cleaning needs to be done."

# Set variables that are common to wad-image* targets.
WI_LEVELS := levels
WI_SCRIPTS := scripts
WI_ALL_OPTIONS := $(WI_OPTIONS) $(if $(WI_BW), --colors-images bw,) \
    $(if $(WI_CMD), --show-cmd $(WI_CMD),) $(if $(WI_GIF), -d gif,) \
    $(if $(WI_NO_SHOW),, -s) $(if $(WI_VERBOSE), -v,)
WI_IMAGES := $(WI_SCRIPTS)/wad2image/images
wad-image-common:

# Generating images for WADs in "levels" directory and show the result.
WI_LATEST := $(shell ls -1t $(WI_LEVELS)/*.wad | head -n 1)
WI_FILES := $(if $(WI_PATT), $(WI_LEVELS)/$(WI_PATT).wad, $(WI_LATEST))
wad-image: wad-image-common
	@echo "Generating images for WADs in \"$(WI_LEVELS)\"."
	scripts/wad2image/bin/wad2image.py $(WI_ALL_OPTIONS) $(WI_FILES)

wad-image-clean: wad-image-common
	rm -rf $(WI_IMAGES)

# Diffing WADs in "levels" using git and show the diff."
wad-image-diff: wad-image-common
	@echo "Diffing WADs in \"$(WI_LEVELS)\" using git."
	scripts/wad2image/integration/git-wad-diff.sh "$(WI_COMMIT)" "$(WI_LEVELS)" $(WI_ALL_OPTIONS)

wad-image-help:
	@echo "Help for wad-image* targets and WI_* variable which can be used to see"
	@echo "differences between WAD revisions, or to simply view WADs. The following targets"
	@echo "depend on wad2image being copied or symlinked to the \"scripts\" directory."
	@echo "Images are created in \"$(WI_IMAGES)\". wad2image can be downloaded"
	@echo "from http://selliott.org/utilities/wad2image."
	@echo ""
	@echo "  Targets:"
	@echo ""
	@echo "    wad-image       Generate generate images for WAD files that are in the"
	@echo "                    workspace."
	@echo "    wad-image-clean Remove \"$(WI_IMAGES)\" as well as all files in"
	@echo "                    it."
	@echo "    wad-image-diff  Use git to generate diff image showing the differences"
	@echo "                    between two revisions of WAD files. By default the"
	@echo "                    difference is between latest HEAD and the workspace, but the"
	@echo "                    WI_COMMIT variable can be used to generate other diffs."
	@echo "    wad-image-help  This help message."
	@echo ""
	@echo "  Variables:"
	@echo ""
	@echo "    WI_BW           Make diff images black or white (high contrast) instead of"
	@echo "                    full color. This applies to wad-image-diff-only."
	@echo "    WI_CMD          Command used to display images. \"display\" is used by"
	@echo "                    default. \"animate\" works well for animated GIFs."
	@echo "    WI_COMMIT       When the wad-image-diff target is invoked this variable"
	@echo "                    specifies which revisions are compared. It's similar to"
	@echo "                    git's \"commit\" argument."
	@echo "    WI_GIF          Create animated GIFs instead of color coded files for the"
	@echo "                    diff."
	@echo "    WI_LEVELS       Subdirectory with the level WADs."
	@echo "    WI_OPTIONS      Additional command line options for wad2image."
	@echo "    WI_NO_SHOW      If set then don't show the images after creating them."
	@echo "    WI_PATT         Files patterns that are applied to files in the \"levels\""
	@echo "                    directory without the \".wad\" suffix. For example,"
	@echo "                    \"map0*\" to get MAP01 - MAP09. This applies to wad-image"
	@echo "                    only."
	@echo "    WI_VERBOSE      If set then make wad2image more verbose."
	@echo ""
	@echo "  Examples:"
	@echo ""
	@echo "    Verbosely create and display an image for the most recently modified WAD"
	@echo "    file in \"levels\":"
	@echo "      make wad-image WI_VERBOSE=t"
	@echo ""
	@echo "    Create and display the image for MAP05:"
	@echo "      make wad-image WI_PATT=map05"
	@echo ""
	@echo "    Verbosely create color coded diffs for changed files in the workspace:"
	@echo "      make wad-image-diff WI_VERBOSE=t"
	@echo ""
	@echo "    Same as the above, but with high contrast black or white images:"
	@echo "      make wad-image-diff WI_VERBOSE=t WI_BW=t"
	@echo ""
	@echo "    Same as above, but use animated GIFs to illustrate the diff instead of"
	@echo "    colors:"
	@echo "      make wad-image-diff WI_VERBOSE=t WI_GIF=t WI_CMD=animate"
	@echo ""
	@echo "    Same as above, but illustrate the diff between two git revisions instead of"
	@echo "    the workspace:"
	@echo "      make wad-image-diff WI_VERBOSE=t WI_GIF=t WI_CMD=animate WI_COMMIT=\"0c004ce~..0c004ce\""
