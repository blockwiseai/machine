import os
import asyncio
import time
import socket
from typing import Tuple, List, Set, Optional

import bittensor as bt
from zeus.base.dendrite import ZeusDendrite
from zeus.data.era5.loaders.cds import Era5CDSLoader
from zeus.data.weatherxm.loader import WeatherXMLoader
from zeus.validator.constants import MechanismType
from zeus.protocol import PredictionSynapse


def check_tcp_port(host: str, port: int, timeout_s: float = 2.0) -> Tuple[bool, str]:
    """
    Check if a single TCP port is reachable by attempting a TCP connect.

    Returns:
        (reachable, message)
    """
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True, "connected"
    except ConnectionRefusedError:
        return False, "refused (host reachable, port closed or actively rejected)"
    except socket.timeout:
        return False, "timeout (filtered/firewalled/routing issue likely)"
    except OSError as e:
        return False, f"oserror: {e}"


class Machine:
    def __init__(self, netuid: int, sleep: int = 12):
        bt.logging.info(f"Starting machine on netuid:{netuid}")
        self.netuid = netuid
        self.subtensor = bt.subtensor("ws://205.172.59.24:9944/")
        self.sleep = sleep

        coldkeypub = os.environ.get(
            "BT_COLDKEYPUB", "5Fh91gqLz8iheXzjAPUXn7rLyrSpDVWirYwkUXYQjwjuKRRo"
        )
        hotkey_mnemonic = os.environ.get(
            "BT_HOTKEY_MNEMONIC",
            "consider dose luxury noodle choice position advance noodle people tiny bright right",
        )

        if not coldkeypub or not hotkey_mnemonic:
            raise RuntimeError(
                "Missing secrets. Set env vars: BT_COLDKEYPUB and BT_HOTKEY_MNEMONIC"
            )

        wallet = bt.wallet()
        wallet.regenerate_coldkeypub(coldkeypub, overwrite=True)
        wallet.regenerate_hotkey(hotkey_mnemonic, overwrite=True)

        self.dendrite = ZeusDendrite(wallet=wallet)

        data_loaders = {
            MechanismType.ERA5: Era5CDSLoader(),
            MechanismType.WEATHER_XM: WeatherXMLoader(),
        }
        self.dl_wxm: WeatherXMLoader = data_loaders[MechanismType.WEATHER_XM]
        self.dl_era5: Era5CDSLoader = data_loaders[MechanismType.ERA5]

        self._portcheck_sem = asyncio.Semaphore(200)
        self._send_sem = asyncio.Semaphore(500)

        # --- Window cache (ONE TIME per window) ---
        self._in_challenge_window: bool = False
        self._window_ips: Optional[Set[str]] = None
        self._window_synapse: Optional[PredictionSynapse] = None

        # Used to detect epoch reset
        self._last_blocks_since_last_step: Optional[int] = None

    def get_blocks_since_last_step(self) -> int:
        return self.subtensor.subnet(self.netuid).blocks_since_last_step

    def _collect_candidate_ips(self) -> Set[str]:
        """
        Collect candidate ip:port pairs from metagraph (sync).
        """
        m = self.subtensor.metagraph(self.netuid)
        set_ips: Set[str] = set()

        for axon, stake in zip(m.axons, m.S):
            if stake >= 10_000.0:
                continue
            set_ips.add(f"{axon.ip}:{axon.port}")

        return set_ips

    async def _port_ok(self, ip: str, port: int) -> bool:
        """
        Run blocking TCP connect in a thread to avoid blocking the event loop.
        """
        async with self._portcheck_sem:
            ok, _ = await asyncio.to_thread(check_tcp_port, ip, port, 2.0)
            return ok

    async def get_ips_async(self) -> Set[str]:
        """
        Filter candidate ips concurrently (port reachability + custom filters).
        """
        candidates = self._collect_candidate_ips()

        tasks: List[Tuple[str, asyncio.Task[bool]]] = []
        for set_ip in candidates:
            ip, port_str = set_ip.split(":")
            port = int(port_str)

            if ip == "0.0.0.0":
                continue
            if 6971 <= port <= 6980:
                continue

            tasks.append((set_ip, asyncio.create_task(self._port_ok(ip, port))))

        results: Set[str] = set()
        for set_ip, task in tasks:
            try:
                if await task:
                    results.add(set_ip)
                    bt.logging.info(f"Ip to send synapse: {set_ip}")
            except Exception as e:
                bt.logging.warning(f"Port check failed for {set_ip}: {e}")

        return results

    def build_synapse_once(self) -> PredictionSynapse:
        """
        Build synapse once for the whole challenge window.
        """
        return self.dl_era5.get_sample().get_synapse()

    async def _send_to_target(
        self, ip: str, port: int, synapse: PredictionSynapse
    ) -> None:
        """
        Send a batch of requests to a single target (ip, port).
        """
        async with self._send_sem:
            axon_to_send = bt.AxonInfo(
                ip_type=4,
                ip=ip,
                port=port,
                version=9001000,
                hotkey="xxxxx",
                coldkey="xxxxx",
            )

            axons = [axon_to_send for _ in range(50)]
            try:
                _responses: List[PredictionSynapse] = await self.dendrite(
                    axons=axons,
                    synapse=synapse,
                    deserialize=False,
                    timeout=0.3,
                )
            except Exception as e:
                bt.logging.warning(f"Send failed to {ip}:{port}: {e}")

    async def send_requests_parallel_using_window_cache(self) -> None:
        """
        Send requests concurrently to each ip:port cached for the current challenge window.
        Assumes window cache is already populated.
        """
        if not self._window_ips or not self._window_synapse:
            bt.logging.warning("Window cache is empty; skipping send.")
            return

        tasks = []
        for set_ip in self._window_ips:
            ip, port_str = set_ip.split(":")
            tasks.append(
                asyncio.create_task(
                    self._send_to_target(ip, int(port_str), self._window_synapse)
                )
            )

        await asyncio.gather(*tasks, return_exceptions=True)

    def _enter_challenge_window(self) -> None:
        """
        Reset window cache when entering challenge window.
        """
        self._in_challenge_window = True
        self._window_ips = None
        self._window_synapse = None

    def _exit_challenge_window(self) -> None:
        """
        Clear cache when leaving challenge window.
        """
        self._in_challenge_window = False
        self._window_ips = None
        self._window_synapse = None

    async def _populate_window_cache_once(self) -> None:
        """
        Populate IP list and synapse exactly once per window.
        """
        if self._window_ips is None:
            bt.logging.info("Fetching IPs once for the whole challenge window...")
            self._window_ips = await self.get_ips_async()

        if self._window_synapse is None:
            bt.logging.info("Building synapse once for the whole challenge window...")
            self._window_synapse = self.build_synapse_once()

    async def run_async(self) -> None:
        """
        Main loop without blocking the event loop.
        """
        while True:
            bsl = self.get_blocks_since_last_step()
            blocks_left = 360 - bsl

            # Detect epoch boundary by blocks_since_last_step decreasing (reset)
            if (
                self._last_blocks_since_last_step is not None
                and bsl < self._last_blocks_since_last_step
            ):
                bt.logging.info("Epoch reset detected; clearing window cache.")
                self._exit_challenge_window()
            self._last_blocks_since_last_step = bsl

            if blocks_left <= 50:
                if not self._in_challenge_window:
                    bt.logging.info("Entering challenge window.")
                    self._enter_challenge_window()

                # Populate cache ONCE for the whole window
                await self._populate_window_cache_once()

                bt.logging.info(
                    "Challenge window: sending requests (parallel, cached IPs + synapse)."
                )
                await self.send_requests_parallel_using_window_cache()

            else:
                if self._in_challenge_window:
                    bt.logging.info("Leaving challenge window; clearing cache.")
                    self._exit_challenge_window()

                bt.logging.info(
                    f"{(blocks_left) * 12 / 60:.2f} min left until next epoch. Sleeping for {self.sleep}s ..."
                )
                await asyncio.sleep(self.sleep)


if __name__ == "__main__":
    bt.logging.set_debug()
    machine = Machine(netuid=18)
    asyncio.run(machine.run_async())
