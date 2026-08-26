# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest import mock

from recursive_conclusion_lab import (
    AdapterRegistry,
    ChatMessage,
    DummyAdapter,
    ExperimentConfig,
    GenerationConfig,
    ModelCapabilities,
    ModelProfile,
    OpenAIResponsesAdapter,
    RecursiveConclusionSession,
    TemperaturePolicy,
    available_provider_names,
    build_adapter,
    build_compare_args_from_config,
    build_parser,
    canonical_provider_name,
    generation_config_with_min_tokens,
    make_experiment_config_from_args,
    model_profile_catalog,
    parse_provider_specs,
    pin_generation_config,
    register_model_profile,
    resolve_model_profile,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "protocol_scripts" / "convergent_protocol.json"
GPT56_MODELS = (
    "gpt-5.6",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
)


class AdapterRegistryTests(unittest.TestCase):
    def test_builtin_registry_names_aliases_and_keyless_construction(self) -> None:
        self.assertEqual(
            available_provider_names(),
            ("openai", "anthropic", "mistral", "gemini", "hf", "dummy"),
        )
        self.assertEqual(canonical_provider_name(" HuggingFace "), "hf")
        self.assertEqual(canonical_provider_name("hugging_face"), "hf")
        self.assertEqual(canonical_provider_name("OPENAI"), "openai")
        self.assertEqual(
            parse_provider_specs(["HuggingFace=demo/model", "OpenAI=gpt-5.6"]),
            [("hf", "demo/model"), ("openai", "gpt-5.6")],
        )

        with self.assertRaisesRegex(ValueError, "Unsupported generation provider"):
            canonical_provider_name("not-registered")
        with self.assertRaisesRegex(ValueError, "must not be blank"):
            canonical_provider_name("  ")

        with mock.patch.dict(os.environ, {}, clear=True):
            adapters = {
                provider: build_adapter(provider, model)
                for provider, model in (
                    ("openai", "gpt-5.6"),
                    ("anthropic", "claude-offline"),
                    ("mistral", "mistral-offline"),
                    ("gemini", "gemini-offline"),
                    ("hf", "hf/offline"),
                )
            }
        self.assertIsInstance(adapters["openai"], OpenAIResponsesAdapter)
        for provider, adapter in adapters.items():
            with self.subTest(provider=provider):
                self.assertEqual(adapter.provider_name, provider)
                self.assertIsNone(adapter.api_key)
        self.assertIsInstance(build_adapter("dummy", "dummy-v1"), DummyAdapter)

    def test_registry_extension_is_lazy_and_rejects_alias_collisions(self) -> None:
        calls: list[str] = []
        registry = AdapterRegistry(adapter_kind="test")

        def factory(model: str) -> dict[str, str]:
            calls.append(model)
            return {"model": model}

        registry.register("future", factory, aliases=("future_alias",))
        self.assertEqual(calls, [])
        self.assertEqual(registry.provider_names(), ("future",))
        self.assertEqual(registry.canonical_name("FUTURE_ALIAS"), "future")
        self.assertEqual(registry.build("future_alias", "model-v1"), {"model": "model-v1"})
        self.assertEqual(calls, ["model-v1"])

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("another", factory, aliases=("future_alias",))


class ModelProfileTests(unittest.TestCase):
    def test_gpt56_exact_models_share_the_frozen_profile(self) -> None:
        for model in GPT56_MODELS:
            with self.subTest(model=model):
                profile = resolve_model_profile("openai", model)
                self.assertEqual(profile.profile_id, "openai.gpt-5.6.v1")
                self.assertEqual(profile.version, 1)
                self.assertEqual(
                    profile.capabilities.reasoning_efforts,
                    ("none", "low", "medium", "high", "xhigh", "max"),
                )
                self.assertEqual(
                    profile.capabilities.temperature_policy,
                    TemperaturePolicy.REASONING_NONE_ONLY,
                )
                self.assertEqual(profile.default_reasoning_effort, "none")
                self.assertEqual(profile.default_reasoning_mode, "standard")
                self.assertEqual(profile.default_reasoning_context, "current_turn")

        for near_match in ("gpt-5.6-pro", "gpt-5.60", "gpt-5.6-cyber"):
            with self.subTest(near_match=near_match):
                profile = resolve_model_profile("openai", near_match)
                self.assertEqual(profile.profile_id, "openai.default")
                self.assertEqual(profile.model_ids, ())

    def test_profile_catalog_and_unique_registration_are_data_driven(self) -> None:
        descriptor = next(
            item
            for item in model_profile_catalog()
            if item["id"] == "openai.gpt-5.6.v1"
        )
        self.assertEqual(descriptor["models"], list(GPT56_MODELS))
        self.assertEqual(
            descriptor["capabilities"]["temperature_policy"],
            "reasoning_none_only",
        )
        self.assertEqual(
            descriptor["defaults"],
            {
                "reasoning_effort": "none",
                "reasoning_mode": "standard",
                "reasoning_context": "current_turn",
                "text_verbosity": None,
            },
        )

        fake_model = "rcl-offline-profile-fixture-v1"
        fake_profile = ModelProfile(
            profile_id="openai.rcl-offline-fixture.v1",
            provider="openai",
            model_ids=(fake_model,),
            capabilities=ModelCapabilities(
                reasoning_modes=("future_mode",),
                temperature_policy=TemperaturePolicy.NEVER,
            ),
        )
        if resolve_model_profile("openai", fake_model).profile_id == "openai.default":
            register_model_profile(fake_profile)
        self.assertEqual(
            resolve_model_profile("openai", fake_model).profile_id,
            fake_profile.profile_id,
        )
        parsed = build_parser().parse_args(
            [
                "repl",
                "--provider",
                "openai",
                "--model",
                fake_model,
                "--reasoning-mode",
                "future_mode",
            ]
        )
        self.assertEqual(parsed.reasoning_mode, "future_mode")
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_model_profile(fake_profile)

        invalid_capabilities = (
            ("uppercase", ModelCapabilities(reasoning_modes=("PRO",))),
            ("reserved", ModelCapabilities(reasoning_modes=("auto",))),
            ("duplicate", ModelCapabilities(reasoning_modes=("pro", "pro"))),
        )
        for suffix, capabilities in invalid_capabilities:
            with self.subTest(suffix=suffix):
                with self.assertRaises(ValueError):
                    register_model_profile(
                        ModelProfile(
                            profile_id=f"openai.invalid-{suffix}.v1",
                            provider="openai",
                            model_ids=(f"invalid-{suffix}-model",),
                            capabilities=capabilities,
                        )
                    )

    def test_generation_configs_pin_resolved_profile_defaults_and_version(self) -> None:
        adapter = OpenAIResponsesAdapter("gpt-5.6-terra")
        pinned = pin_generation_config(
            adapter,
            GenerationConfig(temperature=0.0, max_tokens=220),
        )
        self.assertEqual(pinned.reasoning_effort, "none")
        self.assertEqual(pinned.reasoning_mode, "standard")
        self.assertEqual(pinned.reasoning_context, "current_turn")
        self.assertIsNone(pinned.text_verbosity)
        self.assertEqual(pinned.model_profile_id, "openai.gpt-5.6.v1")
        self.assertEqual(pinned.model_profile_version, 1)

        mismatched = GenerationConfig(
            model_profile_id="openai.gpt-5.6.v1",
            model_profile_version=999,
        )
        with self.assertRaisesRegex(ValueError, "refusing to resume"):
            adapter.resolve_generation_settings(mismatched)

        incomplete = GenerationConfig(model_profile_id="openai.gpt-5.6.v1")
        with self.assertRaisesRegex(ValueError, "requires both"):
            adapter.resolve_generation_settings(incomplete)

    def test_generator_and_observer_profiles_are_pinned_independently(self) -> None:
        config = ExperimentConfig(
            probe_config=GenerationConfig(
                temperature=0.0,
                max_tokens=220,
                reasoning_effort="medium",
                reasoning_mode="pro",
                reasoning_context="all_turns",
                text_verbosity="high",
            )
        )
        RecursiveConclusionSession(
            adapter=build_adapter("openai", "gpt-5.6-sol"),
            observer_adapter=build_adapter("anthropic", "claude-offline"),
            config=config,
        )
        self.assertEqual(config.probe_config.model_profile_id, "openai.gpt-5.6.v1")
        self.assertEqual(config.probe_config.reasoning_effort, "medium")
        self.assertIsNotNone(config.observer_config)
        assert config.observer_config is not None
        self.assertEqual(config.observer_config.model_profile_id, "anthropic.default")
        self.assertIsNone(config.observer_config.reasoning_effort)
        self.assertIsNone(config.observer_config.reasoning_mode)
        self.assertIsNone(config.observer_config.reasoning_context)
        self.assertIsNone(config.observer_config.text_verbosity)


class OpenAIResponsesAdapterTests(unittest.TestCase):
    @staticmethod
    def messages() -> list[ChatMessage]:
        return [
            ChatMessage(role="user", content="First constraint"),
            ChatMessage(role="assistant", content="Provisional answer"),
            ChatMessage(role="user", content="Now conclude"),
        ]

    def test_legacy_openai_payload_is_unchanged(self) -> None:
        adapter = OpenAIResponsesAdapter(
            "gpt-4.1-mini-2025-04-14",
            api_key="offline-key",
        )
        payload = adapter.build_request_payload(
            system="System instruction",
            messages=self.messages(),
            config=GenerationConfig(
                temperature=0.2,
                max_tokens=321,
                timeout_seconds=17,
            ),
        )
        self.assertEqual(
            payload,
            {
                "model": "gpt-4.1-mini-2025-04-14",
                "input": [
                    {"role": "user", "content": "First constraint"},
                    {"role": "assistant", "content": "Provisional answer"},
                    {"role": "user", "content": "Now conclude"},
                ],
                "max_output_tokens": 321,
                "temperature": 0.2,
                "store": False,
                "instructions": "System instruction",
            },
        )
        self.assertNotIn("reasoning", payload)
        self.assertNotIn("text", payload)

    def test_gpt56_defaults_preserve_model_identity_and_zero_temperature(self) -> None:
        for model in GPT56_MODELS:
            with self.subTest(model=model):
                adapter = OpenAIResponsesAdapter(model, api_key="offline-key")
                payload = adapter.build_request_payload(
                    system=None,
                    messages=self.messages(),
                    config=GenerationConfig(temperature=0.0, max_tokens=220),
                )
                self.assertEqual(payload["model"], model)
                self.assertEqual(payload["temperature"], 0.0)
                self.assertEqual(
                    payload["reasoning"],
                    {
                        "effort": "none",
                        "mode": "standard",
                        "context": "current_turn",
                    },
                )
                self.assertNotIn("text", payload)

    def test_gpt56_reasoning_controls_and_temperature_policy(self) -> None:
        adapter = OpenAIResponsesAdapter("gpt-5.6-terra", api_key="offline-key")
        for effort in ("none", "low", "medium", "high", "xhigh", "max"):
            with self.subTest(effort=effort):
                config = GenerationConfig(
                    temperature=0.7,
                    max_tokens=900,
                    reasoning_effort=effort,
                    reasoning_mode="pro",
                    reasoning_context="all_turns",
                    text_verbosity="high",
                )
                payload = adapter.build_request_payload(
                    system="Judge carefully",
                    messages=self.messages(),
                    config=config,
                )
                self.assertEqual(payload["reasoning"]["effort"], effort)
                self.assertEqual(payload["reasoning"]["mode"], "pro")
                self.assertEqual(payload["reasoning"]["context"], "all_turns")
                self.assertEqual(payload["text"], {"verbosity": "high"})
                if effort == "none":
                    self.assertEqual(payload["temperature"], 0.7)
                else:
                    self.assertNotIn("temperature", payload)
                    resolved = adapter.resolve_generation_settings(config)
                    self.assertEqual(resolved.omitted_controls, ("temperature",))

    def test_invalid_or_unsupported_controls_fail_before_transport(self) -> None:
        adapter = OpenAIResponsesAdapter("gpt-5.6", api_key="offline-key")
        invalid_configs = (
            GenerationConfig(reasoning_effort="ultra"),
            GenerationConfig(reasoning_mode="turbo"),
            GenerationConfig(reasoning_context="automatic"),
            GenerationConfig(text_verbosity="extreme"),
        )
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "not supported"):
                    adapter.build_request_payload(
                        system=None,
                        messages=self.messages(),
                        config=config,
                    )

        legacy = OpenAIResponsesAdapter("gpt-4.1-mini-2025-04-14")
        with self.assertRaisesRegex(ValueError, "Supported values: none"):
            legacy.build_request_payload(
                system=None,
                messages=self.messages(),
                config=GenerationConfig(reasoning_effort="low"),
            )

    def test_response_parser_handles_reasoning_text_refusal_and_incomplete_status(self) -> None:
        adapter = OpenAIResponsesAdapter("gpt-5.6", api_key="offline-key")
        data = {
            "id": "resp-offline-1",
            "model": "gpt-5.6-sol",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "First paragraph."},
                        {"type": "refusal", "refusal": "Unused refusal."},
                        {"type": "output_text", "text": "Final paragraph."},
                    ],
                },
            ],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 9,
                "output_tokens_details": {"reasoning_tokens": 4},
            },
            "reasoning": {
                "effort": "low",
                "mode": "standard",
                "context": "current_turn",
                "summary": None,
            },
        }
        response = adapter.parse_response(
            data,
            adapter_metadata={"profile_id": "openai.gpt-5.6.v1"},
        )
        self.assertEqual(response.text, "First paragraph.\nFinal paragraph.")
        self.assertEqual(response.finish_reason, "incomplete:max_output_tokens")
        self.assertEqual(response.request_id, "resp-offline-1")
        self.assertEqual(response.usage, data["usage"])
        self.assertEqual(response.adapter_metadata["response_model"], "gpt-5.6-sol")
        self.assertEqual(response.adapter_metadata["refusal_count"], 1)
        self.assertEqual(
            response.adapter_metadata["response_reasoning"],
            {
                "effort": "low",
                "mode": "standard",
                "context": "current_turn",
            },
        )
        self.assertRegex(response.adapter_metadata["raw_response_sha256"], r"^[0-9a-f]{64}$")

    def test_generate_uses_injected_transport_and_records_sent_profile(self) -> None:
        transport = mock.Mock(
            return_value={
                "id": "resp-offline-2",
                "model": "gpt-5.6-luna",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Done."}],
                    }
                ],
                "usage": {"output_tokens": 5},
            }
        )
        adapter = OpenAIResponsesAdapter(
            "gpt-5.6-luna",
            api_key="offline-key",
            transport=transport,
        )
        response = adapter.generate(
            system="System instruction",
            messages=self.messages(),
            config=GenerationConfig(
                temperature=0.4,
                max_tokens=444,
                timeout_seconds=19,
                reasoning_effort="medium",
                text_verbosity="low",
            ),
        )
        self.assertEqual(response.text, "Done.")
        request = transport.call_args.kwargs
        self.assertEqual(request["url"], OpenAIResponsesAdapter.url)
        self.assertEqual(request["headers"]["Authorization"], "Bearer offline-key")
        self.assertEqual(request["timeout_seconds"], 19)
        self.assertNotIn("temperature", request["payload"])
        self.assertEqual(request["payload"]["reasoning"]["effort"], "medium")
        self.assertEqual(response.adapter_metadata["profile_id"], "openai.gpt-5.6.v1")
        self.assertEqual(response.adapter_metadata["sent"]["max_tokens"], 444)
        self.assertEqual(response.adapter_metadata["omitted_controls"], ["temperature"])

    def test_missing_key_is_deferred_until_generate(self) -> None:
        transport = mock.Mock()
        with mock.patch.dict(os.environ, {}, clear=True):
            adapter = OpenAIResponsesAdapter("gpt-5.6", transport=transport)
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                adapter.generate(
                    system=None,
                    messages=self.messages(),
                    config=GenerationConfig(),
                )
        transport.assert_not_called()


class ModelControlCliTests(unittest.TestCase):
    def test_probe_budget_copy_preserves_model_controls(self) -> None:
        original = GenerationConfig(
            temperature=0.0,
            max_tokens=220,
            timeout_seconds=45,
            reasoning_effort="none",
            reasoning_mode="standard",
            reasoning_context="current_turn",
            text_verbosity="low",
            model_profile_id="openai.gpt-5.6.v1",
            model_profile_version=1,
        )
        expanded = generation_config_with_min_tokens(original, 480)
        self.assertEqual(expanded.max_tokens, 480)
        self.assertEqual(expanded.temperature, 0.0)
        self.assertEqual(expanded.timeout_seconds, 45)
        self.assertEqual(expanded.reasoning_effort, "none")
        self.assertEqual(expanded.reasoning_mode, "standard")
        self.assertEqual(expanded.reasoning_context, "current_turn")
        self.assertEqual(expanded.text_verbosity, "low")
        self.assertEqual(expanded.model_profile_id, "openai.gpt-5.6.v1")
        self.assertEqual(expanded.model_profile_version, 1)
        self.assertEqual(original.max_tokens, 220)

    def test_cli_propagates_reply_and_probe_controls(self) -> None:
        args = build_parser().parse_args(
            [
                "compare",
                "--script",
                str(SCRIPT_PATH),
                "--providers",
                "openai=gpt-5.6-sol",
                "huggingface=demo/model",
                "--reasoning-effort",
                "medium",
                "--reasoning-mode",
                "pro",
                "--reasoning-context",
                "all_turns",
                "--text-verbosity",
                "high",
                "--probe-reasoning-effort",
                "low",
                "--probe-reasoning-mode",
                "standard",
                "--probe-reasoning-context",
                "current_turn",
                "--probe-text-verbosity",
                "low",
                "--observer-reasoning-effort",
                "high",
                "--observer-reasoning-mode",
                "standard",
                "--observer-reasoning-context",
                "current_turn",
                "--observer-text-verbosity",
                "medium",
            ]
        )
        self.assertEqual(
            parse_provider_specs(args.providers),
            [("openai", "gpt-5.6-sol"), ("hf", "demo/model")],
        )
        config = make_experiment_config_from_args(args)
        self.assertEqual(config.reply_config.reasoning_effort, "medium")
        self.assertEqual(config.reply_config.reasoning_mode, "pro")
        self.assertEqual(config.reply_config.reasoning_context, "all_turns")
        self.assertEqual(config.reply_config.text_verbosity, "high")
        self.assertEqual(config.probe_config.temperature, 0.0)
        self.assertEqual(config.probe_config.reasoning_effort, "low")
        self.assertEqual(config.probe_config.reasoning_mode, "standard")
        self.assertEqual(config.probe_config.reasoning_context, "current_turn")
        self.assertEqual(config.probe_config.text_verbosity, "low")
        self.assertIsNotNone(config.observer_config)
        assert config.observer_config is not None
        self.assertEqual(config.observer_config.reasoning_effort, "high")
        self.assertEqual(config.observer_config.reasoning_mode, "standard")
        self.assertEqual(config.observer_config.reasoning_context, "current_turn")
        self.assertEqual(config.observer_config.text_verbosity, "medium")

    def test_json_config_and_arm_override_propagate_controls(self) -> None:
        args = build_compare_args_from_config(
            {
                "script": str(SCRIPT_PATH),
                "providers": ["openai=gpt-5.6-terra"],
                "args": {
                    "reasoning_effort": "low",
                    "reasoning_mode": "standard",
                    "reasoning_context": "current_turn",
                    "text_verbosity": "medium",
                    "probe_reasoning_effort": "none",
                    "probe_text_verbosity": "low",
                },
            },
            args_override={
                "reasoning_effort": "max",
                "probe_reasoning_context": "all_turns",
            },
            arm_name="profile_override",
        )
        config = make_experiment_config_from_args(args)
        self.assertEqual(args.arm_name, "profile_override")
        self.assertEqual(config.reply_config.reasoning_effort, "max")
        self.assertEqual(config.reply_config.reasoning_mode, "standard")
        self.assertEqual(config.reply_config.reasoning_context, "current_turn")
        self.assertEqual(config.reply_config.text_verbosity, "medium")
        self.assertEqual(config.probe_config.reasoning_effort, "none")
        self.assertEqual(config.probe_config.reasoning_context, "all_turns")
        self.assertEqual(config.probe_config.text_verbosity, "low")


if __name__ == "__main__":
    unittest.main()
