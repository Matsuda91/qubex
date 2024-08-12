from __future__ import annotations

from enum import Enum
from typing import Final, Literal, Union

from pydantic.dataclasses import dataclass

from .model import Model

CLOCK_MASTER_ADDRESS: Final = "10.3.0.255"

DEFAULT_LO_FREQ: Final = 9_000_000_000
DEFAULT_CNCO_FREQ: Final = 1_500_000_000
DEFAULT_FNCO_FREQ: Final = 0
DEFAULT_VATT: Final = 3072  # 0xC00
DEFAULT_FULLSCALE_CURRENT: Final = 40527
DEFAULT_NDELAY: Final = 7
DEFAULT_NWAIT: Final = 0


class BoxType(Enum):
    QUEL1_A = "quel1-a"
    QUEL1_B = "quel1-b"
    QUBE_RIKEN_A = "qube-riken-a"
    QUBE_RIKEN_B = "qube-riken-b"
    QUBE_OU_A = "qube-ou-a"
    QUBE_OU_B = "qube-ou-b"


class PortType(Enum):
    NA = "NA"
    READ_IN = "READ_IN"
    READ_OUT = "READ_OUT"
    CTRL = "CTRL"
    PUMP = "PUMP"
    MNTR_IN = "MNTR_IN"
    MNTR_OUT = "MNTR_OUT"


PORT_DIRECTION: Final = {
    PortType.READ_IN: "in",
    PortType.READ_OUT: "out",
    PortType.CTRL: "out",
    PortType.PUMP: "out",
    PortType.MNTR_IN: "in",
    PortType.MNTR_OUT: "out",
}


PORT_MAPPING: Final = {
    BoxType.QUEL1_A: {
        0: PortType.READ_IN,
        1: PortType.READ_OUT,
        2: PortType.CTRL,
        3: PortType.PUMP,
        4: PortType.CTRL,
        5: PortType.MNTR_IN,
        6: PortType.MNTR_OUT,
        7: PortType.READ_IN,
        8: PortType.READ_OUT,
        9: PortType.CTRL,
        10: PortType.PUMP,
        11: PortType.CTRL,
        12: PortType.MNTR_IN,
        13: PortType.MNTR_OUT,
    },
    BoxType.QUEL1_B: {
        0: PortType.NA,
        1: PortType.CTRL,
        2: PortType.CTRL,
        3: PortType.CTRL,
        4: PortType.CTRL,
        5: PortType.MNTR_IN,
        6: PortType.MNTR_OUT,
        7: PortType.NA,
        8: PortType.CTRL,
        9: PortType.CTRL,
        10: PortType.CTRL,
        11: PortType.CTRL,
        12: PortType.MNTR_IN,
        13: PortType.MNTR_OUT,
    },
    BoxType.QUBE_RIKEN_A: {
        0: PortType.READ_OUT,
        1: PortType.READ_IN,
        2: PortType.PUMP,
        3: PortType.MNTR_OUT,
        4: PortType.MNTR_IN,
        5: PortType.CTRL,
        6: PortType.CTRL,
        7: PortType.CTRL,
        8: PortType.CTRL,
        9: PortType.MNTR_IN,
        10: PortType.MNTR_OUT,
        11: PortType.PUMP,
        12: PortType.READ_IN,
        13: PortType.READ_OUT,
    },
    BoxType.QUBE_RIKEN_B: {
        0: PortType.CTRL,
        1: PortType.NA,
        2: PortType.CTRL,
        3: PortType.MNTR_OUT,
        4: PortType.MNTR_IN,
        5: PortType.CTRL,
        6: PortType.CTRL,
        7: PortType.CTRL,
        8: PortType.CTRL,
        9: PortType.MNTR_IN,
        10: PortType.MNTR_OUT,
        11: PortType.CTRL,
        12: PortType.NA,
        13: PortType.CTRL,
    },
    BoxType.QUBE_OU_A: {
        0: PortType.READ_OUT,
        1: PortType.READ_IN,
        2: PortType.PUMP,
        3: PortType.NA,
        4: PortType.NA,
        5: PortType.CTRL,
        6: PortType.CTRL,
        7: PortType.CTRL,
        8: PortType.CTRL,
        9: PortType.NA,
        10: PortType.NA,
        11: PortType.PUMP,
        12: PortType.READ_IN,
        13: PortType.READ_OUT,
    },
    BoxType.QUBE_OU_B: {
        0: PortType.CTRL,
        1: PortType.NA,
        2: PortType.CTRL,
        3: PortType.NA,
        4: PortType.NA,
        5: PortType.CTRL,
        6: PortType.CTRL,
        7: PortType.CTRL,
        8: PortType.CTRL,
        9: PortType.NA,
        10: PortType.NA,
        11: PortType.CTRL,
        12: PortType.NA,
        13: PortType.CTRL,
    },
}

NUMBER_OF_CHANNELS: Final = {
    BoxType.QUEL1_A: {
        0: 4,
        1: 1,
        2: 3,
        3: 1,
        4: 3,
        5: 4,
        6: 1,
        7: 4,
        8: 1,
        9: 3,
        10: 1,
        11: 3,
        12: 4,
        13: 1,
    },
    BoxType.QUEL1_B: {
        0: 0,
        1: 1,
        2: 3,
        3: 1,
        4: 3,
        5: 4,
        6: 1,
        7: 0,
        8: 1,
        9: 3,
        10: 1,
        11: 3,
        12: 4,
        13: 1,
    },
    BoxType.QUBE_RIKEN_A: {
        0: 1,
        1: 4,
        2: 1,
        3: 1,
        4: 4,
        5: 3,
        6: 3,
        7: 3,
        8: 3,
        9: 4,
        10: 1,
        11: 1,
        12: 4,
        13: 1,
    },
    BoxType.QUBE_RIKEN_B: {
        0: 1,
        1: 0,
        2: 1,
        3: 1,
        4: 4,
        5: 3,
        6: 3,
        7: 3,
        8: 3,
        9: 4,
        10: 1,
        11: 1,
        12: 0,
        13: 1,
    },
    BoxType.QUBE_OU_A: {
        0: 1,
        1: 4,
        2: 1,
        3: 0,
        4: 0,
        5: 3,
        6: 3,
        7: 3,
        8: 3,
        9: 0,
        10: 0,
        11: 1,
        12: 4,
        13: 1,
    },
    BoxType.QUBE_OU_B: {
        0: 1,
        1: 0,
        2: 1,
        3: 0,
        4: 0,
        5: 3,
        6: 3,
        7: 3,
        8: 3,
        9: 0,
        10: 0,
        11: 1,
        12: 0,
        13: 1,
    },
}


def create_ports(
    box_id: str,
    box_type: BoxType,
) -> list[Union[GenPort, CapPort]]:
    ports: list[Union[GenPort, CapPort]] = []
    port_index = {
        PortType.NA: 0,
        PortType.READ_IN: 0,
        PortType.READ_OUT: 0,
        PortType.CTRL: 0,
        PortType.PUMP: 0,
        PortType.MNTR_IN: 0,
        PortType.MNTR_OUT: 0,
    }
    for port_num, port_type in PORT_MAPPING[box_type].items():
        index = port_index[port_type]
        if port_type == PortType.NA:
            port_id = f"{box_id}.NA{index}"
        elif port_type == PortType.READ_IN:
            port_id = f"{box_id}.READ{index}.IN"
        elif port_type == PortType.READ_OUT:
            port_id = f"{box_id}.READ{index}.OUT"
        elif port_type == PortType.CTRL:
            port_id = f"{box_id}.CTRL{index}"
        elif port_type == PortType.PUMP:
            port_id = f"{box_id}.PUMP{index}"
        elif port_type == PortType.MNTR_IN:
            port_id = f"{box_id}.MNTR{index}.IN"
        elif port_type == PortType.MNTR_OUT:
            port_id = f"{box_id}.MNTR{index}.OUT"
        else:
            raise ValueError(f"Invalid port type: {port_type}")
        n_channels = NUMBER_OF_CHANNELS[box_type].get(port_num, 0)
        port: Union[GenPort, CapPort]
        if port_type in (PortType.READ_IN, PortType.MNTR_IN):
            port = CapPort(
                id=port_id,
                box_id=box_id,
                number=port_num,
                type=port_type,
                channels=[
                    CapChannel(
                        id=f"{port_id}{channel_num}",
                        port_id=port_id,
                        number=channel_num,
                    )
                    for channel_num in range(n_channels)
                ],
            )
        elif port_type in (PortType.READ_OUT, PortType.MNTR_OUT):
            port = GenPort(
                id=port_id,
                box_id=box_id,
                number=port_num,
                type=port_type,
                sideband="U",
                channels=[
                    GenChannel(
                        id=f"{port_id}{channel_num}",
                        port_id=port_id,
                        number=channel_num,
                    )
                    for channel_num in range(n_channels)
                ],
            )
        elif port_type in (PortType.CTRL, PortType.PUMP):
            port = GenPort(
                id=port_id,
                box_id=box_id,
                number=port_num,
                type=port_type,
                sideband="L",
                channels=[
                    GenChannel(
                        id=f"{port_id}.CH{channel_num}",
                        port_id=port_id,
                        number=channel_num,
                    )
                    for channel_num in range(n_channels)
                ],
            )
        else:
            raise ValueError(f"Invalid port type: {port_type}")
        ports.append(port)
        port_index[port_type] += 1
    return ports


@dataclass
class Box(Model):
    id: str
    name: str
    type: BoxType
    address: str
    adapter: str
    ports: list[Union[GenPort, CapPort]]

    @classmethod
    def new(
        cls,
        *,
        id: str,
        name: str,
        type: BoxType | str,
        address: str,
        adapter: str,
    ) -> Box:
        type = BoxType(type) if isinstance(type, str) else type
        return cls(
            id=id,
            name=name,
            type=type,
            address=address,
            adapter=adapter,
            ports=create_ports(id, type),
        )

    @property
    def input_ports(self) -> list[CapPort]:
        return [port for port in self.ports if isinstance(port, CapPort)]

    @property
    def output_ports(self) -> list[GenPort]:
        return [port for port in self.ports if isinstance(port, GenPort)]

    @property
    def control_ports(self) -> list[Port]:
        return [port for port in self.ports if port.is_control_port]

    @property
    def readout_ports(self) -> list[Port]:
        return [port for port in self.ports if port.is_readout_port]

    @property
    def monitor_ports(self) -> list[Port]:
        return [port for port in self.ports if port.is_monitor_port]

    @property
    def pump_ports(self) -> list[Port]:
        return [port for port in self.ports if port.is_pump_port]


@dataclass
class Port(Model):
    id: str
    box_id: str
    number: int
    type: PortType
    channels: Union[list[GenChannel], list[CapChannel]]

    @property
    def direction(self) -> str:
        return PORT_DIRECTION[self.type]

    @property
    def is_input_port(self) -> bool:
        return self.direction == "in"

    @property
    def is_output_port(self) -> bool:
        return self.direction == "out"

    @property
    def is_control_port(self) -> bool:
        return self.type == PortType.CTRL

    @property
    def is_readout_port(self) -> bool:
        return self.type in (PortType.READ_IN, PortType.READ_OUT)

    @property
    def is_monitor_port(self) -> bool:
        return self.type in (PortType.MNTR_IN, PortType.MNTR_OUT)

    @property
    def is_pump_port(self) -> bool:
        return self.type == PortType.PUMP


@dataclass
class GenPort(Port):
    channels: list[GenChannel]
    sideband: Literal["U", "L"]
    lo_freq: int = DEFAULT_LO_FREQ
    cnco_freq: int = DEFAULT_CNCO_FREQ
    vatt: int = DEFAULT_VATT
    fullscale_current: int = DEFAULT_FULLSCALE_CURRENT
    rfswitch: str = "pass"


@dataclass
class CapPort(Port):
    channels: list[CapChannel]
    lo_freq: int = DEFAULT_LO_FREQ
    cnco_freq: int = DEFAULT_CNCO_FREQ
    rfswitch: str = "open"


@dataclass
class Channel(Model):
    id: str
    port_id: str
    number: int


@dataclass
class GenChannel(Channel):
    fnco_freq: int = DEFAULT_FNCO_FREQ
    nwait: int = DEFAULT_NWAIT


@dataclass
class CapChannel(Channel):
    fnco_freq: int = DEFAULT_FNCO_FREQ
    ndelay: int = DEFAULT_NDELAY


class QubeSystem:
    def __init__(
        self,
        *,
        boxes: list[Box],
        clock_master_address: str = CLOCK_MASTER_ADDRESS,
    ):
        self._clock_master_address: Final = clock_master_address
        self._box_dict: Final = {box.id: box for box in boxes}

    @property
    def clock_master_address(self) -> str:
        return self._clock_master_address

    @property
    def boxes(self) -> list[Box]:
        return list(self._box_dict.values())

    def get_box(self, box_id: str) -> Box:
        try:
            return self._box_dict[box_id]
        except KeyError:
            raise KeyError(f"Box `{box_id}` not found.")
