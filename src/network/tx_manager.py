"""Transaction broadcast ordering for transport-layer workers.

The manager in this module ensures that sequence assignment is thread-isolated
per account, allowing parallel signing and dispatch without global contention.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, MutableMapping, Optional, Protocol

from src.network.nonce_tracker import nonce_tracker

logger = logging.getLogger(__name__)

Payload = MutableMapping[str, Any]
Signer = Callable[[Payload], Payload]
Dispatcher = Callable[[Payload], Any]


class TxPayloadSigner(Protocol):
    def __call__(self, payload: Payload) -> Payload:
        """Sign and return a transaction payload."""


class TxPayloadDispatcher(Protocol):
    def __call__(self, payload: Payload) -> Any:
        """Dispatch a signed transaction payload."""


@dataclass
class BroadcastResult:
    """Result wrapper that exposes the assigned sequence for tracking."""

    account_id: str
    sequence: int
    payload: Payload
    dispatch_result: Any


class TxManager:
    """Coordinate transaction signing and dispatch using a thread-isolated nonce tracker."""

    def __init__(self, sequence_field: str = "sequence") -> None:
        self.sequence_field = sequence_field

    def broadcast(
        self,
        account_id: str,
        payload: Payload,
        *,
        signer: Signer,
        dispatcher: Dispatcher,
        seed_sequence: Optional[int] = None,
    ) -> BroadcastResult:
        """Assign a sequence, sign the payload, and dispatch it.

        Sequence assignment is handled by the nonce_tracker to eliminate
        contention. Signing and dispatching occur outside the critical
        section to allow parallel processing across accounts.

        Args:
            account_id: Source account whose transaction sequence is tracked.
            payload: Transaction payload. It is deep-copied before mutation.
            signer: Callable that signs the sequenced payload and returns it.
            dispatcher: Callable that sends the signed payload to the network.
            seed_sequence: Required on first use for an account.

        Returns:
            BroadcastResult containing the assigned sequence, signed payload,
            and dispatcher response.
        """

        if not account_id:
            raise ValueError("account_id is required.")

        # Claim sequence slot independently via thread-isolated nonce tracker.
        sequence = nonce_tracker.get_next_nonce(account_id, seed=seed_sequence)

        # Perform signing and dispatch outside the lock to maximize throughput.
        sequenced_payload = self._with_sequence(payload, sequence)
        signed_payload = signer(sequenced_payload)
        self._assert_signed_sequence(signed_payload, sequence)
        dispatch_result = dispatcher(signed_payload)

        logger.info(
            "[TxManager] Dispatched transaction for %s with sequence %d",
            account_id,
            sequence,
        )

        return BroadcastResult(
            account_id=account_id,
            sequence=sequence,
            payload=signed_payload,
            dispatch_result=dispatch_result,
        )

    def sync_sequence(self, account_id: str, sequence: int) -> None:
        """Set an account counter to a known-good sequence value."""
        nonce_tracker.sync_nonce(account_id, sequence)
        logger.info("[TxManager] Synced sequence for %s to %d", account_id, sequence)

    def invalidate(self, account_id: Optional[str] = None) -> None:
        """Clear one account sequence or all tracked account sequences."""
        nonce_tracker.invalidate(account_id)
        if account_id:
            logger.info("[TxManager] Invalidated sequence for %s", account_id)
        else:
            logger.info("[TxManager] Invalidated all tracked sequences")

    def current_sequence(self, account_id: str) -> Optional[int]:
        """Return the cached sequence for an account, if seeded."""
        # Accessing internal state of nonce_tracker to get current sequence.
        # Note: NonceTracker doesn't have a public current_sequence method,
        # we may need to add it or access the _nonces dict carefully.
        return nonce_tracker._nonces.get(account_id)

    def _with_sequence(self, payload: Payload, sequence: int) -> Payload:
        sequenced_payload = copy.deepcopy(dict(payload))
        sequenced_payload[self.sequence_field] = sequence
        return sequenced_payload

    def _assert_signed_sequence(self, payload: Payload, sequence: int) -> None:
        if payload.get(self.sequence_field) != sequence:
            raise ValueError(
                "Signer returned a payload with a mismatched sequence value."
            )


tx_manager = TxManager()

__all__ = [
    "BroadcastResult",
    "TxManager",
    "tx_manager",
]
