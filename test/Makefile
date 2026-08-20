# Makefile
# See https://docs.cocotb.org/en/stable/quickstart.html for more info

# defaults
SIM ?= icarus
FST ?= -fst # Use more efficient FST format
TOPLEVEL_LANG ?= verilog
SRC_DIR = $(PWD)/../src
PROJECT_SOURCES = tt_um_kush1434_proof.v proof_core.v accumulator.v mac_serial.v

ifneq ($(GATES),yes)

# RTL simulation:
SIM_BUILD				= sim_build/rtl
VERILOG_SOURCES += $(addprefix $(SRC_DIR)/,$(PROJECT_SOURCES))

else

# Gate level simulation:
SIM_BUILD				= sim_build/gl
COMPILE_ARGS    += -DGL_TEST
COMPILE_ARGS    += -DFUNCTIONAL
COMPILE_ARGS    += -DSIM
VERILOG_SOURCES += $(PDK_ROOT)/ihp-sg13g2/libs.ref/sg13g2_io/verilog/sg13g2_io.v
VERILOG_SOURCES += $(PDK_ROOT)/ihp-sg13g2/libs.ref/sg13g2_stdcell/verilog/sg13g2_stdcell.v

# this gets copied in by the GDS action workflow
VERILOG_SOURCES += $(PWD)/gate_level_netlist.v

# SDF back-annotation. Off unless SDF_FILE is set, so `GATES=yes make` -- what
# the tt-gds-action gl_test job runs -- is untouched and stays the functional
# check it has always been.
#
# SDF_FILE must be a file that has been through sdf_prep.py: the raw OpenSTA
# output aborts Icarus's SDF reader part-way and silently leaves most of the
# netlist at zero delay.
#
# -gspecify   Icarus omits specify blocks by default, and with them
#             $sdf_annotate. It says so once, at compile time, and then
#             simulates happily at zero delay.
# -ginterconnect  without it every INTERCONNECT entry is an error, and the
#             errors abort the rest of the file.
#
# Delays only. Icarus implements no timing checks in any version, so setup and
# hold are still STA's word alone -- see sdf_prep.py.
ifneq ($(SDF_FILE),)
SIM_BUILD				= sim_build/gl_sdf
GL_IN_DELAY_NS  ?= 1.0
COMPILE_ARGS    += -gspecify -ginterconnect
COMPILE_ARGS    += -DUSE_SDF -DGL_IN_DELAY_NS=$(GL_IN_DELAY_NS)
# COCOTB_PLUSARGS, not PLUSARGS: cocotb 2.0 (pinned in requirements.txt) still
# accepts the old name but prints a deprecation warning for it.
COCOTB_PLUSARGS += +sdf=$(SDF_FILE)
endif

endif

# Allow sharing configuration between design and testbench via `include`:
COMPILE_ARGS 		+= -I$(SRC_DIR)

# Include the testbench sources:
VERILOG_SOURCES += $(PWD)/tb.v
TOPLEVEL = tb

# List test modules to run, separated by commas and without the .py suffix:
# `?=` so the SDF job can run test_sdf instead. The default is unchanged, so
# "43 top-level tests" still means what it has always meant.
COCOTB_TEST_MODULES ?= test

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim
