"""
TTP (Template Text Parser) 示例
从思科 / 华为设备配置及终端输出中提取结构化数据

TTP 工作原理：
  1. 编写一段"模板"，模板里用 {{ 变量名 }} 标记要捕获的字段
  2. 把原始文本 + 模板喂给 TTP，输出 JSON/dict 结果
"""

from ttp import ttp
import json


def parse(template: str, text: str, title: str) -> None:
    parser = ttp(data=text, template=template)
    parser.parse()
    result = parser.result(format="json")[0]
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)
    print(json.dumps(json.loads(result), indent=2, ensure_ascii=False))


# ─────────────────────────────────────────────────────────────
# 场景 1：思科路由器 —— show ip interface brief
# ─────────────────────────────────────────────────────────────
cisco_interfaces_text = """
Interface              IP-Address      OK? Method Status                Protocol
GigabitEthernet0/0     192.168.1.1     YES NVRAM  up                    up
GigabitEthernet0/1     10.0.0.1        YES NVRAM  up                    up
GigabitEthernet0/2     unassigned      YES NVRAM  administratively down down
Loopback0              1.1.1.1         YES NVRAM  up                    up
""".strip()

cisco_interfaces_template = """
<group name="interfaces">
{{ interface | re("[A-Za-z0-9/]+") }} {{ ip }} {{ ok }} {{ method }} {{ status | PHRASE }} {{ protocol }}
</group>
"""

parse(cisco_interfaces_template, cisco_interfaces_text, "思科 show ip interface brief")


# ─────────────────────────────────────────────────────────────
# 场景 2：思科路由器 —— 接口配置段落
# ─────────────────────────────────────────────────────────────
cisco_config_text = """
interface GigabitEthernet0/0
 description WAN-Link-to-ISP
 ip address 203.0.113.1 255.255.255.0
 duplex auto
 speed auto
 no shutdown
!
interface GigabitEthernet0/1
 description LAN-Core-Switch
 ip address 192.168.10.1 255.255.255.0
 duplex full
 speed 1000
 no shutdown
!
interface Loopback0
 description MGMT
 ip address 10.255.255.1 255.255.255.255
 no shutdown
!
""".strip()

cisco_config_template = """
<group name="interfaces">
interface {{ name }}
 description {{ description | PHRASE }}
 ip address {{ ip }} {{ mask }}
 duplex {{ duplex }}
 speed {{ speed }}
</group>
"""

parse(cisco_config_template, cisco_config_text, "思科接口配置段落解析")


# ─────────────────────────────────────────────────────────────
# 场景 3：思科 —— show ip route
# ─────────────────────────────────────────────────────────────
cisco_route_text = """
Codes: C - connected, S - static, R - RIP, O - OSPF

O    10.1.0.0/24 [110/20] via 192.168.1.254, 00:03:12, GigabitEthernet0/0
O    10.2.0.0/24 [110/30] via 192.168.1.254, 00:03:12, GigabitEthernet0/0
S    0.0.0.0/0 [1/0] via 203.0.113.254
C    192.168.1.0/24 is directly connected, GigabitEthernet0/0
""".strip()

cisco_route_template = """
<group name="routes">
{{ proto }} {{ network }}/{{ prefix_len }} [{{ ad }}/{{ metric }}] via {{ nexthop }}, {{ age }}, {{ interface }}
{{ proto }} {{ network }}/{{ prefix_len }} [{{ ad }}/{{ metric }}] via {{ nexthop }}
{{ proto }} {{ network }}/{{ prefix_len }} is directly connected, {{ interface }}
</group>
"""

parse(cisco_route_template, cisco_route_text, "思科 show ip route")


# ─────────────────────────────────────────────────────────────
# 场景 4：华为路由器 —— display ip interface brief
# ─────────────────────────────────────────────────────────────
huawei_interface_text = """
*down: administratively down
^down: standby
(l): loopback
(s): spoofing
The number of interface that is UP in Physical is 3
The number of interface that is DOWN in Physical is 1

Interface                         IP Address/Mask      Physical   Protocol
GigabitEthernet0/0/0              10.1.1.1/24          up         up
GigabitEthernet0/0/1              192.168.0.1/24       up         up
GigabitEthernet0/0/2              unassigned           *down      down
LoopBack0                         1.1.1.1/32           up         up(s)
""".strip()

huawei_interface_template = """
<group name="interfaces">
{{ name | re("[A-Za-z0-9/]+") }} {{ ip_mask }} {{ physical }} {{ protocol }}
</group>
"""

parse(huawei_interface_template, huawei_interface_text, "华为 display ip interface brief")


# ─────────────────────────────────────────────────────────────
# 场景 5：华为 —— 接口配置段落
# ─────────────────────────────────────────────────────────────
huawei_config_text = """
#
interface GigabitEthernet0/0/0
 description uplink-to-core
 ip address 10.0.0.2 255.255.255.0
 ospf enable 1 area 0.0.0.0
#
interface GigabitEthernet0/0/1
 description access-vlan100
 ip address 172.16.1.1 255.255.255.128
#
interface LoopBack0
 ip address 10.255.0.1 255.255.255.255
#
""".strip()

huawei_config_template = """
<group name="interfaces">
interface {{ name }}
 description {{ description | PHRASE }}
 ip address {{ ip }} {{ mask }}
 ospf enable {{ ospf_process }} area {{ ospf_area }}
</group>
"""

parse(huawei_config_template, huawei_config_text, "华为接口配置段落解析")


# ─────────────────────────────────────────────────────────────
# 场景 6：华为 —— display bgp peer
# ─────────────────────────────────────────────────────────────
huawei_bgp_text = """
 BGP local router ID : 10.255.0.1
 Local AS number : 65001

 Total number of peers : 3        Peers in established state : 2

  Peer            V          AS  MsgRcvd  MsgSent  OutQ  Up/Down       State  PrefRcv
  10.0.0.1        4       65002     1024     1020     0  1d02h         Established    8
  10.0.1.1        4       65003      512      510     0  10:30:22      Established   12
  10.0.2.1        4       65004        0        0     0  00:00:00      Active          0
""".strip()

huawei_bgp_template = """
<group name="bgp_info">
 BGP local router ID : {{ router_id }}
 Local AS number : {{ local_as }}
</group>

<group name="bgp_peers">
  {{ peer_ip | re("\\d+\\.\\d+\\.\\d+\\.\\d+") }} {{ version }} {{ remote_as }} {{ msg_rcvd }} {{ msg_sent }} {{ outq }} {{ updown }} {{ state }} {{ prefrcv }}
</group>
"""

parse(huawei_bgp_template, huawei_bgp_text, "华为 display bgp peer")
