"""Pythonic ACS (Action Code Script) generation."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from doomgen.utils.paths import get_bundled_compiler, validate_executable


class ScriptType(Enum):
    """Types of ACS scripts."""
    OPEN = auto()       # Runs when map opens
    ENTER = auto()      # Runs when player enters
    RESPAWN = auto()    # Runs when player respawns
    DEATH = auto()      # Runs when player dies
    LIGHTNING = auto()  # Runs periodically for lightning
    UNLOADING = auto()  # Runs when map unloads
    DISCONNECT = auto() # Runs when player disconnects
    RETURN = auto()     # Runs when returning to map
    EVENT = auto()      # Custom event script
    KILL = auto()       # Runs when thing is killed
    REOPEN = auto()     # Runs when returning to hub
    VOID = auto()       # Callable script (void)


class VariableScope(Enum):
    """Scopes for ACS variables."""
    MAP = auto()        # Visible to all scripts in the map
    WORLD = auto()      # Visible across hub (reset on hub entry)
    GLOBAL = auto()     # Visible across all maps (persistent)


@dataclass
class ACSVariable:
    """
    Represents a variable in ACS.
    """
    name: str
    var_type: str
    scope: VariableScope
    initial: Any = None
    index: int | None = None

    def to_code(self) -> str:
        """Generate ACS code for this variable declaration."""
        if self.scope == VariableScope.MAP:
            # int map_var = 0;
            line = f"{self.var_type} {self.name}"
            if self.initial is not None:
                line += f" = {self.initial}"
            return line + ";"

        elif self.scope == VariableScope.WORLD:
            # world int 1:world_var;
            if self.index is None:
                raise ValueError(f"World variable '{self.name}' must have an index.")
            return f"world {self.var_type} {self.index}:{self.name};"

        elif self.scope == VariableScope.GLOBAL:
            # global int 1:global_var;
            if self.index is None:
                raise ValueError(f"Global variable '{self.name}' must have an index.")
            return f"global {self.var_type} {self.index}:{self.name};"

        return ""


@dataclass
class ACSScript:
    """
    Represents a single ACS script.

    Attributes:
        number: Script number (1-999).
        script_type: Type of script trigger.
        body: C-style script body code.
        args: Script arguments (for ENTER scripts).
    """
    number: int
    script_type: ScriptType
    body: str
    args: list[str] = field(default_factory=list)

    def to_code(self) -> str:
        """Generate ACS code for this script."""
        type_str = {
            ScriptType.OPEN: "OPEN",
            ScriptType.ENTER: "ENTER",
            ScriptType.RESPAWN: "RESPAWN",
            ScriptType.DEATH: "DEATH",
            ScriptType.LIGHTNING: "LIGHTNING",
            ScriptType.UNLOADING: "UNLOADING",
            ScriptType.DISCONNECT: "DISCONNECT",
            ScriptType.RETURN: "RETURN",
            ScriptType.EVENT: "EVENT",
            ScriptType.KILL: "KILL",
            ScriptType.REOPEN: "REOPEN",
            ScriptType.VOID: "",
        }.get(self.script_type, "")

        # Build argument list
        if self.args:
            args_str = ", ".join(f"int {arg}" for arg in self.args)
            header = f"script {self.number} ({args_str})"
        elif type_str:
            header = f"script {self.number} {type_str}"
        else:
            header = f"script {self.number} (void)"

        return f"{header}\n{{\n{self._indent(self.body)}\n}}"

    def _indent(self, code: str, level: int = 1) -> str:
        """Indent code lines."""
        indent = "    " * level
        lines = code.strip().split("\n")
        return "\n".join(indent + line for line in lines)


class ACSBuilder:
    """
    Builder for ACS scripts.

    Provides a Pythonic interface for constructing ACS scripts
    that can be compiled and inserted into WAD files.

    Attributes:
        scripts: List of ACS scripts.
        includes: List of include directives.
        defines: Dictionary of #define macros.
        global_vars: List of global variables.
    """

    def __init__(self):
        """Initialize the ACS builder."""
        self.scripts: list[ACSScript] = []
        self.includes: list[str] = []
        self.defines: dict[str, Any] = {}
        self.variables: list[ACSVariable] = []
        self.global_code_blocks: list[str] = []
        self._next_script_num = 1

    def add_global_code(self, code: str) -> ACSBuilder:
        """
        Add a block of raw global code (e.g. function definitions).
        """
        self.global_code_blocks.append(code)
        return self

    def add_include(self, filename: str) -> ACSBuilder:
        """
        Add an include directive.

        Args:
            filename: File to include (e.g., "zcommon.acs").

        Returns:
            Self for chaining.
        """
        self.includes.append(filename)
        return self

    def add_define(self, name: str, value: Any) -> ACSBuilder:
        """
        Add a #define macro.

        Args:
            name: Macro name.
            value: Macro value.

        Returns:
            Self for chaining.
        """
        self.defines[name] = value
        return self

    def add_map_var(
        self,
        name: str,
        var_type: str = "int",
        initial: Any = None
    ) -> ACSBuilder:
        """
        Add a map-scope variable.
        Visible to all scripts in the current map.

        Args:
            name: Variable name.
            var_type: Variable type ("int", "str", "bool").
            initial: Initial value.
        """
        self.variables.append(ACSVariable(name, var_type, VariableScope.MAP, initial=initial))
        return self

    def add_world_var(
        self,
        name: str,
        index: int,
        var_type: str = "int"
    ) -> ACSBuilder:
        """
        Add a world-scope variable.
        Visible across a hub of maps. Reset when entering a new hub.

        Args:
            name: Variable name.
            index: Unique index (1-256).
            var_type: Variable type.
        """
        self.variables.append(ACSVariable(name, var_type, VariableScope.WORLD, index=index))
        return self

    def add_global_var(
        self,
        name: str,
        index: int,
        var_type: str = "int"
    ) -> ACSBuilder:
        """
        Add a global-scope variable.
        Visible across all maps and persistent.

        Args:
            name: Variable name.
            index: Unique index (1-64).
            var_type: Variable type.
        """
        self.variables.append(ACSVariable(name, var_type, VariableScope.GLOBAL, index=index))
        return self

    def add_script(
        self,
        script_type: ScriptType,
        body: str,
        number: int | None = None,
        args: list[str] | None = None
    ) -> int:
        """
        Add a new script.

        Args:
            script_type: Type of script trigger.
            body: Script body (C-style code).
            number: Script number (auto-assigned if None).
            args: Script arguments.

        Returns:
            The script number.
        """
        if number is None:
            number = self._next_script_num
            self._next_script_num += 1
        else:
            self._next_script_num = max(self._next_script_num, number + 1)

        script = ACSScript(
            number=number,
            script_type=script_type,
            body=body,
            args=args or []
        )

        self.scripts.append(script)
        return number

    def define_door_logic(
        self,
        tag: int,
        open_time: int = 35,  # Tics (35 = 1 second)
        close_time: int = 105,  # Tics until auto-close
        speed: int = 16
    ) -> int:
        """
        Create a door open/close script.

        Args:
            tag: Sector tag for the door.
            open_time: Time to stay open (in tics, 35 = 1 second).
            close_time: Time until auto-close.
            speed: Door movement speed.

        Returns:
            Script number.
        """
        body = f"""
Door_Open({tag}, {speed});
Delay({open_time});
Door_Close({tag}, {speed});
"""
        return self.add_script(ScriptType.OPEN, body.strip())

    def define_teleport_logic(
        self,
        source_tag: int,
        dest_tag: int
    ) -> int:
        """
        Create a teleporter script.

        Args:
            source_tag: Source sector tag.
            dest_tag: Destination thing tag.

        Returns:
            Script number.
        """
        body = f"""
Teleport({dest_tag}, 0, 0);
"""
        return self.add_script(ScriptType.ENTER, body.strip())

    def define_spawn_enemies(
        self,
        thing_type: int,
        spawn_tag: int,
        count: int = 1,
        delay: int = 35
    ) -> int:
        """
        Create an enemy spawn script.

        Args:
            thing_type: Doom thing type to spawn.
            spawn_tag: Thing tag for spawn location.
            count: Number to spawn.
            delay: Delay between spawns (tics).

        Returns:
            Script number.
        """
        body = f"""
for (int i = 0; i < {count}; i++)
{{
    SpawnSpot("{thing_type}", {spawn_tag});
    Delay({delay});
}}
"""
        return self.add_script(ScriptType.OPEN, body.strip())

    def define_message(
        self,
        message: str,
        script_type: ScriptType = ScriptType.ENTER
    ) -> int:
        """
        Create a message display script.

        Args:
            message: Message to display.
            script_type: When to trigger.

        Returns:
            Script number.
        """
        # Escape quotes in message
        escaped = message.replace('"', '\\"')
        body = f'Print(s:"{escaped}");'
        return self.add_script(script_type, body)

    def define_sector_light(
        self,
        tag: int,
        min_light: int = 64,
        max_light: int = 255,
        pulse_time: int = 35
    ) -> int:
        """
        Create a pulsing light effect.

        Args:
            tag: Sector tag.
            min_light: Minimum light level.
            max_light: Maximum light level.
            pulse_time: Pulse duration (tics).

        Returns:
            Script number.
        """
        body = f"""
while (true)
{{
    Light_Fade({tag}, {max_light}, {pulse_time});
    Delay({pulse_time});
    Light_Fade({tag}, {min_light}, {pulse_time});
    Delay({pulse_time});
}}
"""
        return self.add_script(ScriptType.OPEN, body.strip())

    def to_code(self) -> str:
        """
        Generate complete ACS source code.

        Returns:
            ACS source code as a string.
        """
        lines = []

        # Includes
        for inc in self.includes:
            lines.append(f'#include "{inc}"')

        if self.includes:
            lines.append("")

        # Defines
        for name, value in self.defines.items():
            lines.append(f"#define {name} {value}")

        if self.defines:
            lines.append("")

        # Variables
        for var in self.variables:
            lines.append(var.to_code())

        if self.variables:
            lines.append("")

        # Global Code Blocks
        for block in self.global_code_blocks:
            lines.append(block)
            lines.append("")

        # Scripts
        for script in self.scripts:
            lines.append(script.to_code())
            lines.append("")

        return "\n".join(lines)

    def compile(
        self,
        output_path: str | Path | None = None,
        acc_path: str | None = None
    ) -> bytes | None:
        """
        Compile the ACS scripts to bytecode.

        Requires the ACC compiler to be installed and accessible.
        If no acc_path is provided, uses ACC compiler in ./acc/

        Args:
            output_path: Path for compiled output (temporary if None).
            acc_path: Path to ACC compiler executable.

        Returns:
            Compiled bytecode as bytes, or None if compilation fails.

        Raises:
            FileNotFoundError: If ACC compiler is not found.
            subprocess.CalledProcessError: If compilation fails.
        """
        if acc_path is None:
            acc_path = get_bundled_compiler("acc")

        source_code = self.to_code()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write source
            source_file = Path(tmpdir) / "script.acs"
            source_file.write_text(source_code)

            # Output file
            if output_path:
                obj_file = Path(output_path)
            else:
                obj_file = Path(tmpdir) / "script.o"

            # Validate ACC executable exists
            if not validate_executable(acc_path):
                raise FileNotFoundError(
                    f"ACC compiler not found at '{acc_path}'. "
                    "Ensure ACC is installed or available via bundled ./acc directory."
                )

            # Compile
            try:
                result = subprocess.run(
                    [acc_path, str(source_file), str(obj_file)],
                    capture_output=True,
                    text=True,
                    check=True
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"ACC compiler not found at '{acc_path}'. "
                    "Please install ACC and add it to your PATH."
                )

            # Read bytecode
            if obj_file.exists():
                return obj_file.read_bytes()

            return None

    def save_source(self, filepath: str | Path) -> None:
        """
        Save ACS source code to a file.

        Args:
            filepath: Output file path.
        """
        Path(filepath).write_text(self.to_code())
