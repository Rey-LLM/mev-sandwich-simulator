"""Layer 15: localization (Russian by default, English kept as the reference language).

Every human-facing string that Layer 15 adds or re-labels goes through this
module: the comments of ``sandwich.toml``, the browser settings editor, the
telemetry window (:mod:`simulator.gui`), the interactive transaction graph
(:mod:`simulator.tx_graph`), the on-disk code factory's file headers and logs
(:mod:`simulator.code_factory`), the HTML charts and the CLI help.

Machine-readable output (JSON report keys, TOML keys, CLI flag names, rule
ids) stays in English on purpose: scripts and tests depend on it, and a
translated key would silently break them.

Language selection, highest priority first: an explicit ``lang`` argument,
``$MEV_SIM_LANG``, ``[ui] language`` in ``sandwich.toml``, then
:data:`DEFAULT_LANGUAGE` (``"ru"``).

The catalog is a flat mapping ``key -> {"ru": ..., "en": ...}``. Templates
use :meth:`str.format` fields; :func:`catalog_issues` checks that both
languages exist for every key, that both use the same fields and that a
Russian text is actually Russian (it contains Cyrillic, or is on the short
list of technical tokens that are written the same way in both languages).
"""

from __future__ import annotations

import os
import re
import string
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

LANGUAGES: Tuple[str, ...] = ("ru", "en")
DEFAULT_LANGUAGE = "ru"
LANG_ENV_VAR = "MEV_SIM_LANG"

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
# Strings that are legitimately identical in Russian (units, tickers, protocol names).
SAME_IN_BOTH: Tuple[str, ...] = ("ETH", "BTC", "WEI", "wei", "gwei", "sat", "sat/vB", "PBS", "MEV", "CPU", "OK",
                                 "JSON", "HTML", "TOML", "UTXO", "ERC-20", "RBF", "CPFP", "MAX Brutal Simulation")


def normalize_language(value: Any) -> str:
    """``"ru"`` / ``"en"`` from common spellings; raises ValueError for anything else."""
    text = str(value or "").strip().lower().replace("_", "-")
    aliases = {"ru": "ru", "rus": "ru", "russian": "ru", "ru-ru": "ru", "русский": "ru",
               "en": "en", "eng": "en", "english": "en", "en-us": "en", "en-gb": "en"}
    if text in aliases:
        return aliases[text]
    raise ValueError(f"unknown language {value!r}; expected one of {', '.join(LANGUAGES)}")


def resolve_language(lang: Optional[str] = None, configured: Optional[str] = None) -> str:
    """The language to use: ``lang`` > ``$MEV_SIM_LANG`` > ``configured`` > :data:`DEFAULT_LANGUAGE`."""
    for candidate in (lang, os.environ.get(LANG_ENV_VAR), configured):
        if candidate:
            try:
                return normalize_language(candidate)
            except ValueError:
                continue
    return DEFAULT_LANGUAGE


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

CATALOG: Dict[str, Dict[str, str]] = {
    # -- run modes ----------------------------------------------------------------------------------------------
    "mode.moderate": {"en": "Moderate Scan", "ru": "Умеренное сканирование"},
    "mode.brutal": {"en": "Brutal Aggressive Simulation", "ru": "Жёсткая агрессивная симуляция"},
    "mode.max": {"en": "MAX Brutal Simulation", "ru": "MAX Brutal Simulation (максимальная нагрузка)"},
    "mode_note.moderate": {
        "en": "Moderate Scan: read-only telemetry, gas-limit audit, passive dataflow (50-60% of CPUs).",
        "ru": "Умеренное сканирование: телеметрия только для чтения, аудит лимита газа, пассивный поток "
              "данных (50-60% ядер CPU)."},
    "mode_note.brutal": {
        "en": "Brutal Aggressive Simulation: multi-block sandwich search on the capped pool (50-60% of CPUs).",
        "ru": "Жёсткая агрессивная симуляция: многоблочный поиск сэндвичей на ограниченном пуле процессов "
              "(50-60% ядер CPU)."},
    "mode_note.max": {
        "en": "MAX Brutal Simulation: no CPU cap (100% of logical CPUs), dense shards, policy-gradient path fuzz, "
              "AI agent factory.",
        "ru": "MAX Brutal Simulation: без ограничения CPU (100% логических ядер), плотные шарды, фаззинг путей "
              "градиента политики, фабрика ИИ-агентов."},
    # -- sandwich.toml header -------------------------------------------------------------------------------------
    "toml.header": {
        "en": "# sandwich.toml: central settings for the MEV sandwich simulator.\n"
              "#\n"
              "# Precedence, lowest first: the built-in preset of the mode < this file\n"
              "# < $MEV_SIM_WORKERS (worker count only) < --set TABLE.KEY=VALUE < flags.\n"
              "#\n"
              "#   python -m simulator --show-config      print the effective settings\n"
              "#   python -m simulator --sync-config      add keys that newer versions introduce\n"
              "#   python -m simulator --save-config ...  write command-line values back here\n"
              "#   python -m simulator --gui              edit this file in the browser\n",
        "ru": "# sandwich.toml: центральный файл настроек симулятора MEV-сэндвичей.\n"
              "#\n"
              "# Приоритет, от низшего к высшему: встроенный пресет режима < этот файл\n"
              "# < $MEV_SIM_WORKERS (только число процессов) < --set ТАБЛИЦА.КЛЮЧ=ЗНАЧЕНИЕ < флаги.\n"
              "#\n"
              "#   python -m simulator --show-config      показать действующие настройки\n"
              "#   python -m simulator --sync-config      добавить ключи, появившиеся в новых версиях\n"
              "#   python -m simulator --save-config ...  записать значения из командной строки в файл\n"
              "#   python -m simulator --gui              редактировать этот файл в браузере\n"},
    "toml.schema_version": {"en": "layout version (written by the simulator)",
                            "ru": "версия разметки (записывается симулятором)"},
    "toml.effective": {"en": "Effective settings: {mode}, {workers} worker(s) under the {policy} CPU policy.",
                       "ru": "Действующие настройки: {mode}, рабочих процессов: {workers}, политика CPU {policy}."},
    # -- table titles -----------------------------------------------------------------------------------------------
    "table.meta": {"en": "Meta", "ru": "Служебное"},
    "table.run": {"en": "Run", "ru": "Запуск"},
    "table.parallel": {"en": "Parallel", "ru": "Параллелизм"},
    "table.gas": {"en": "Gas", "ru": "Газ"},
    "table.search": {"en": "Sandwich search", "ru": "Поиск сэндвичей"},
    "table.pbs": {"en": "PBS builders", "ru": "Билдеры PBS"},
    "table.private_relay": {"en": "Private relay", "ru": "Приватный релей"},
    "table.rl": {"en": "Reinforcement learning", "ru": "Обучение с подкреплением"},
    "table.trace": {"en": "Trace output", "ru": "Трассировка"},
    "table.audit": {"en": "Integrity audit", "ru": "Аудит целостности"},
    "table.patterns": {"en": "MEV pattern store", "ru": "Хранилище MEV-паттернов"},
    "table.ui": {"en": "Interface", "ru": "Интерфейс"},
    "table.modes": {"en": "Run matrices", "ru": "Матрицы режимов"},
    # -- settings editor ------------------------------------------------------------------------------------------
    "editor.title": {"en": "Simulator settings", "ru": "Настройки симулятора"},
    "editor.saved_as": {"en": "saved as {digest}", "ru": "сохранено, дайджест {digest}"},
    "editor.not_created": {"en": "not created yet (saving creates it)",
                           "ru": "файл ещё не создан (сохранение создаст его)"},
    "editor.not_saved": {"en": "Not saved.", "ru": "Не сохранено."},
    "editor.saved": {"en": "Saved.", "ru": "Сохранено."},
    "editor.setting": {"en": "Setting", "ru": "Параметр"},
    "editor.save": {"en": "Save", "ru": "Сохранить"},
    # -- CLI ----------------------------------------------------------------------------------------------------------
    "cli.description": {
        "en": "Offline MEV sandwich simulator. Settings come from the mode preset, sandwich.toml, "
              "$MEV_SIM_WORKERS, --set and the flags below, in that order.",
        "ru": "Офлайн-симулятор MEV-сэндвичей. Настройки берутся из пресета режима, sandwich.toml, "
              "$MEV_SIM_WORKERS, --set и флагов ниже, именно в таком порядке."},
    "cli.error": {"en": "simulator: error: {message}", "ru": "simulator: ошибка: {message}"},
    "cli.warning": {"en": "simulator: warning: {message}", "ru": "simulator: предупреждение: {message}"},
    "cli.group.layer15": {"en": "Layer 15: code factory, transaction graph, telemetry",
                          "ru": "Слой 15: фабрика кода, граф транзакций, телеметрия"},
    "cli.code_factory": {"en": "write bot source files to disk, deploy them in arena heats and retrain the generator",
                         "ru": "записать исходники ботов на диск, развернуть их в заездах арены и дообучить "
                               "генератор"},
    "cli.tx_graph": {"en": "write the interactive BTC / ETH transaction graph (HTML) to FILE",
                     "ru": "записать интерактивный граф транзакций BTC / ETH (HTML) в FILE"},
    "cli.monitor": {"en": "open the telemetry window (needs tkinter)",
                    "ru": "открыть окно телеметрии (нужен tkinter)"},
    "cli.monitor_html": {"en": "record telemetry while the code factory runs and write an HTML dashboard to FILE",
                         "ru": "записать телеметрию во время работы фабрики кода и сохранить HTML-панель в FILE"},
    "cli.group.layer16": {"en": "Layer 16: sandwich detector on real blocks and pre-trade check",
                          "ru": "Слой 16: детектор сэндвичей в реальных блоках и проверка свопа перед отправкой"},
    "cli.detect_file": {"en": "find sandwiches in a saved block export (receipts+logs or call traces, .json or .json.gz)",
                        "ru": "найти сэндвичи в сохранённом экспорте блока (квитанции с логами или трейсы, "
                              ".json или .json.gz)"},
    "cli.detect_block": {"en": "find sandwiches in mainnet block N (or 'latest') fetched over --rpc",
                         "ru": "найти сэндвичи в блоке N мейннета (или 'latest'), загруженном через --rpc"},
    "cli.detect_count": {"en": "with --detect-block: scan K consecutive blocks ending at that block (default 1)",
                         "ru": "с --detect-block: просканировать K блоков подряд, заканчивая указанным (по умолчанию 1)"},
    "cli.save_export": {"en": "with --detect-block: also save the fetched block data to FILE for offline reuse",
                        "ru": "с --detect-block: сохранить загруженные данные блока в FILE для офлайн-анализа"},
    "cli.rpc": {"en": "Ethereum JSON-RPC endpoint (default $ETH_RPC_URL)",
                "ru": "JSON-RPC узел Ethereum (по умолчанию $ETH_RPC_URL)"},
    "cli.check_swap": {"en": "check a planned swap described in a JSON file: is it sandwichable, what slippage is safe",
                       "ru": "проверить планируемый своп из JSON-файла: можно ли его засэндвичить и какой "
                             "слиппедж безопасен"},
    "cli.check_pool": {"en": "check a planned swap on a live pool ADDRESS (with --rpc, --token-in, --amount-in)",
                       "ru": "проверить планируемый своп на живом пуле ADDRESS (с --rpc, --token-in, --amount-in)"},
    "cli.pool_version": {"en": "pool type for --check-pool (default v2)",
                         "ru": "тип пула для --check-pool (по умолчанию v2)"},
    "cli.token_in": {"en": "input token address for --check-pool",
                     "ru": "адрес входного токена для --check-pool"},
    "cli.amount_in": {"en": "input amount in raw token units for --check-pool",
                      "ru": "сумма на входе в минимальных единицах токена для --check-pool"},
    "cli.slippage_bps": {"en": "slippage tolerance you plan to sign, in bps (default 50)",
                         "ru": "допуск слиппеджа, который вы собираетесь подписать, в б.п. (по умолчанию 50)"},
    "cli.gas_price": {"en": "attacker gas price in wei for the pre-trade check (default 0 = ignore gas)",
                      "ru": "цена газа атакующего в wei для проверки свопа (по умолчанию 0 = без учёта газа)"},
    "cli.eth_price": {"en": "input-token units per 1 ETH, to count gas when the input is not WETH",
                      "ru": "сколько единиц входного токена стоит 1 ETH, чтобы учесть газ, если вход не WETH"},
    "guard.advice.safe": {"en": "Not sandwichable at {slippage} bps: no front-run pays for itself.",
                          "ru": "При {slippage} б.п. своп не засэндвичить: ни один фронтран не окупается."},
    "guard.advice.unsafe": {"en": "Sandwichable: the best attack nets {profit} (input-token units) and costs you "
                                  "{loss} of output.",
                            "ru": "Своп можно засэндвичить: лучшая атака приносит {profit} (в единицах входного "
                                  "токена), а вы теряете {loss} на выходе."},
    "guard.advice.lower_slippage": {"en": "Set slippage to {safe} bps or less.",
                                    "ru": "Поставьте слиппедж не больше {safe} б.п."},
    "guard.advice.split": {"en": "Or split the trade: up to {size} per swap is safe at this slippage (about {parts} swaps).",
                           "ru": "Или разбейте сделку: до {size} за своп безопасно при этом слиппедже "
                                 "(число свопов: примерно {parts})."},
    "guard.advice.private_rpc": {"en": "Send it through a private RPC (Flashbots Protect, MEV Blocker) so it never "
                                       "reaches the public mempool.",
                                 "ru": "Отправьте через приватный RPC (Flashbots Protect, MEV Blocker), чтобы "
                                       "транзакция не попала в публичный мемпул."},
    "guard.note.gas_ignored": {"en": "Gas was ignored (input is not WETH and no --eth-price): the answer is conservative.",
                               "ru": "Газ не учтён (вход не WETH и не задан --eth-price): ответ консервативный."},
    "guard.note.single_range": {"en": "v3 pool without ticks: active liquidity is assumed to continue past the current "
                                      "range (exact for swaps that do not cross a tick).",
                                "ru": "Пул v3 без тиков: активная ликвидность считается неизменной за пределами "
                                      "текущего диапазона (точно для свопов, не пересекающих тик)."},
    "cli.bot_dir": {"en": "directory for generated bots (default $MEV_SIM_BOT_DIR, else src/simulator/generated_bots)",
                    "ru": "каталог сгенерированных ботов (по умолчанию $MEV_SIM_BOT_DIR, иначе "
                          "src/simulator/generated_bots)"},
    "cli.factory_bots": {"en": "bots per code factory generation (default 16)",
                         "ru": "ботов в поколении фабрики кода (по умолчанию 16)"},
    "cli.factory_generations": {"en": "code factory generations (default 3)",
                                "ru": "поколений фабрики кода (по умолчанию 3)"},
    "cli.lang": {"en": "language of human-facing output: ru | en", "ru": "язык вывода для людей: ru | en"},
    # -- code factory -------------------------------------------------------------------------------------------------
    "factory.py_header": {
        "en": "Generated bot {name} (generation {generation}). Simulation only: it runs on the simulator's "
              "in-memory pools and never connects to a network.",
        "ru": "Сгенерированный бот {name} (поколение {generation}). Только для симуляции: работает на пулах "
              "симулятора в памяти и никогда не подключается к сети."},
    "factory.sol_header": {
        "en": "Parameter record of simulated bot {name} (generation {generation}). No executable logic: "
              "constants only.",
        "ru": "Запись параметров симулируемого бота {name} (поколение {generation}). Исполняемой логики нет: "
              "только константы."},
    "factory.fitness": {"en": "fitness: P&L {pnl} wei, wins {wins}", "ru": "приспособленность: P&L {pnl} wei, "
                                                                          "побед {wins}"},
    "factory.log.generation": {
        "en": "generation {generation}: wrote {files} files ({bytes} bytes) in {seconds} s, best P&L {best} wei, "
              "loss {loss_before} -> {loss_after}",
        "ru": "поколение {generation}: записано файлов {files} ({bytes} байт) за {seconds} с, лучший P&L {best} wei, "
              "потери {loss_before} -> {loss_after}"},
    "factory.log.rejected": {"en": "rejected generated file {path}: {reason}",
                             "ru": "сгенерированный файл {path} отклонён: {reason}"},
    "factory.log.deployed": {"en": "deployed {bots} bots from disk into {heats} builder heats",
                             "ru": "развёрнуто ботов с диска: {bots}, заездов билдера: {heats}"},
    # -- transaction graph --------------------------------------------------------------------------------------------
    "graph.title": {"en": "Transaction graph", "ru": "Граф транзакций"},
    "graph.subtitle": {"en": "Simulated BTC (UTXO model) and ETH (account and ERC-20 flows) dataflows",
                       "ru": "Смоделированные потоки BTC (модель UTXO) и ETH (счета и потоки ERC-20)"},
    "graph.tab.eth": {"en": "Ethereum", "ru": "Эфириум"},
    "graph.tab.btc": {"en": "Bitcoin", "ru": "Биткоин"},
    "graph.anomalies": {"en": "Anomalies", "ru": "Аномалии"},
    "graph.no_anomalies": {"en": "No anomalies", "ru": "Аномалий нет"},
    "graph.nodes": {"en": "nodes", "ru": "узлов"},
    "graph.edges": {"en": "edges", "ru": "рёбер"},
    "graph.layers": {"en": "layers", "ru": "слоёв"},
    "graph.crossings": {"en": "edge crossings", "ru": "пересечений рёбер"},
    "graph.help": {"en": "Wheel: zoom. Drag: pan. Click a node: highlight its upstream and downstream flow.",
                   "ru": "Колесо: масштаб. Перетаскивание: сдвиг. Щелчок по узлу: подсветка входящего и "
                         "исходящего потока."},
    "graph.reset": {"en": "Reset view", "ru": "Сбросить вид"},
    "graph.kind.tx": {"en": "transaction", "ru": "транзакция"},
    "graph.kind.account": {"en": "account", "ru": "счёт"},
    "graph.kind.pool": {"en": "pool", "ru": "пул"},
    "graph.kind.builder": {"en": "block builder", "ru": "билдер блока"},
    "graph.kind.burn": {"en": "burned base fee", "ru": "сожжённая базовая комиссия"},
    "graph.kind.bundle": {"en": "evicted bundle", "ru": "вытесненный бандл"},
    "graph.kind.utxo_tx": {"en": "UTXO transaction", "ru": "UTXO-транзакция"},
    "graph.kind.miner": {"en": "miner fees", "ru": "комиссии майнеру"},
    "graph.kind.address": {"en": "address", "ru": "адрес"},
    "anomaly.conservation": {"en": "value not conserved: {diff} {unit} off", "ru": "нарушено сохранение стоимости: "
                                                                                  "расхождение {diff} {unit}"},
    "anomaly.state_root": {"en": "replayed state root differs from the sealed block",
                           "ru": "корень состояния после повтора не совпадает с запечатанным блоком"},
    "anomaly.sandwich": {"en": "sandwich around {victim}: attacker {attacker} gained {profit} wei of {token}",
                         "ru": "сэндвич вокруг {victim}: атакующий {attacker} получил {profit} wei {token}"},
    "anomaly.competition": {"en": "builder competition on {victim}: winner ahead by {margin} wei over {rivals} "
                                  "rival bundle(s)",
                            "ru": "конкуренция билдера за {victim}: победитель впереди на {margin} wei, "
                                  "соперничающих бандлов {rivals}"},
    "anomaly.photo_finish": {"en": "photo finish on {victim}: winner ahead by only {margin} wei",
                             "ru": "фотофиниш за {victim}: победитель впереди всего на {margin} wei"},
    "anomaly.evicted": {"en": "bundle {bundle} evicted: {reason}", "ru": "бандл {bundle} вытеснен: {reason}"},
    "anomaly.double_spend": {"en": "outpoint {outpoint} spent by {count} transactions",
                             "ru": "выход {outpoint} тратят транзакций: {count}"},
    "anomaly.rbf_ok": {"en": "{replacement} replaces {original} (BIP 125: +{fee_delta} sat fee)",
                       "ru": "{replacement} заменяет {original} (BIP 125: комиссия +{fee_delta} sat)"},
    "anomaly.rbf_bad": {"en": "{replacement} cannot replace {original}: {rule}",
                        "ru": "{replacement} не может заменить {original}: {rule}"},
    "anomaly.negative_fee": {"en": "{tx} spends {missing} sat more than its inputs",
                             "ru": "{tx} тратит на {missing} sat больше, чем его входы"},
    "anomaly.missing_input": {"en": "{tx} spends unknown outpoint {outpoint}",
                              "ru": "{tx} тратит неизвестный выход {outpoint}"},
    "anomaly.dust": {"en": "{tx} creates a dust output of {value} sat", "ru": "{tx} создаёт пылевой выход {value} sat"},
    "anomaly.cpfp": {"en": "{child} pays for parent {parent}: package rate {rate} sat/vB",
                     "ru": "{child} оплачивает родителя {parent}: ставка пакета {rate} sat/vB"},
    "rbf.not_signaled": {"en": "original does not signal replaceability", "ru": "оригинал не сигнализирует "
                                                                                 "замену"},
    "rbf.fee_too_low": {"en": "absolute fee below the replaced fee plus relay fee",
                        "ru": "абсолютная комиссия ниже заменяемой плюс комиссия ретрансляции"},
    "rbf.rate_too_low": {"en": "fee rate not higher than the original", "ru": "ставка комиссии не выше оригинала"},
    # -- telemetry GUI ------------------------------------------------------------------------------------------------
    "gui.title": {"en": "MEV simulator telemetry", "ru": "Телеметрия симулятора MEV"},
    "gui.cpu": {"en": "CPU load per core", "ru": "Загрузка CPU по ядрам"},
    "gui.cpu_total": {"en": "Total CPU load", "ru": "Общая загрузка CPU"},
    "gui.synthesis": {"en": "Code synthesis speed", "ru": "Скорость синтеза кода"},
    "gui.convergence": {"en": "Learning convergence", "ru": "Сходимость обучения"},
    "gui.loss": {"en": "loss", "ru": "потери"},
    "gui.loss_before": {"en": "loss before training", "ru": "потери до обучения"},
    "gui.loss_after": {"en": "loss after training", "ru": "потери после обучения"},
    "gui.pnl": {"en": "best P&L", "ru": "лучший P&L"},
    "gui.files_per_s": {"en": "files/s", "ru": "файлов/с"},
    "gui.kib_per_s": {"en": "KiB/s", "ru": "КиБ/с"},
    "gui.core": {"en": "core {index}", "ru": "ядро {index}"},
    "gui.start_factory": {"en": "Run code factory", "ru": "Запустить фабрику кода"},
    "gui.start_max": {"en": "MAX Brutal Simulation stress", "ru": "Стресс MAX Brutal Simulation"},
    "gui.build_graph": {"en": "Build transaction graph", "ru": "Построить граф транзакций"},
    "gui.stop": {"en": "Stop", "ru": "Остановить"},
    "gui.idle": {"en": "Idle", "ru": "Ожидание"},
    "gui.running": {"en": "Running: {task}", "ru": "Выполняется: {task}"},
    "gui.done": {"en": "Done: {task}", "ru": "Готово: {task}"},
    "gui.failed": {"en": "Failed: {task}: {error}", "ru": "Ошибка: {task}: {error}"},
    "gui.no_tk": {"en": "tkinter is not available in this Python; use --monitor-html FILE instead",
                  "ru": "tkinter недоступен в этом Python; используйте --monitor-html FILE"},
    "gui.dashboard": {"en": "Telemetry dashboard", "ru": "Панель телеметрии"},
    "gui.generation": {"en": "generation", "ru": "поколение"},
    "gui.time_s": {"en": "time, s", "ru": "время, с"},
    "gui.ceiling": {"en": "ceiling {value}%", "ru": "потолок {value}%"},
    "gui.samples": {"en": "{count} samples", "ru": "замеров: {count}"},
    "gui.no_data": {"en": "no data yet", "ru": "данных пока нет"},
    "gui.log": {"en": "Log", "ru": "Журнал"},
}

# Russian help text of every setting in config_sync.SETTINGS, by attribute.
SETTING_HELP_RU: Dict[str, str] = {
    "mode": "режим запуска: moderate | brutal | max",
    "seed": "зерно всех детерминированных потоков",
    "native_token": "токен, в котором считаются прибыль и газ",
    "default_tolerance_bps": "допуск проскальзывания (bps) для транзакций без своего допуска",
    "start_method": "способ запуска процессов multiprocessing",
    "parallel_threshold": "пакеты меньше этого размера выполняются в текущем процессе",
    "base_fee_wei": "начальная базовая комиссия (wei)",
    "priority_fee_wei": "приоритетная комиссия сёрчера (wei)",
    "block_gas_limit": "лимит газа блока, приводится к [21000, 30000000]",
    "max_frontrun_bps": "наибольший фронтран в bps от входного резерва",
    "builder_margin_bps": "доля стоимости блока, которую оставляет билдер (bps)",
    "bribe_bps": "доля прибыли сэндвича, выплачиваемая билдеру (bps)",
    "mev_share_refund_bps": "возврат пользователям через MEV-Share (bps)",
    "rl_regime": "рыночный режим RL: calm | volatile | extreme",
    "trace": "записывать трассу транзакций каждого запечатанного блока",
    "trace_path": "файл трассировки (пусто = stderr)",
    "audit": "запускать проверку целостности перед симуляцией",
    "audit_fail_on": "минимальная серьёзность находки, блокирующая запуск: error | warning | info",
    "audit_code_scan": "включить сканирование исходного кода",
    "audit_tick_samples": "число тиков в проверках арифметики тиков",
    "pattern_store": "каталог хранилища MEV-паттернов (пусто = только синтетические эталонные паттерны)",
    "pattern_max_storage_gb": "предел объёма хранилища паттернов в ГиБ (сегменты + индекс)",
    "pattern_segment_mb": "размер ротации файла-сегмента паттернов (МиБ)",
    "pattern_cache_mb": "бюджет LRU-кэша декодированных паттернов (МиБ)",
    "language": "язык интерфейса, журналов и HTML-графиков: ru | en",
    "workers": "рабочие процессы (0 = автоматически по политике CPU)",
    "cpu_policy": "capped = 50-60% ядер CPU, full = 100% логических ядер",
    "blocks": "число симулируемых блоков (слотов)",
    "txs_per_block": "ожидающих транзакций на блок",
    "frontrun_grid": "размеров фронтрана, перебираемых для каждой жертвы",
    "refine_iterations": "шагов уточнения тернарным поиском",
    "fuzz_iterations": "свопов фаззинга инвариантов",
    "layout_fuzz_iterations": "раскладок блока, фаззимых на слот",
    "private_flow_bps": "доля пользовательских свопов, отправляемых в приватный релей (bps)",
    "router_max_hops": "самый длинный маршрут / цикл (1-6)",
    "rl_rounds": "раундов параллельной итерации политики",
    "rl_steps_per_shard": "шагов RL на шард и раунд",
    "rl_eval_steps": "шагов жадной оценки RL",
    "shard_density": "множитель всех фиксированных чисел шардов / чанков",
    "pg_paths": "путей градиента политики, фаззимых параллельно",
    "pg_path_steps": "шагов на путь градиента политики",
    "pg_batch": "путей на задачу рабочего процесса",
    "agent_bots": "ИИ-ботов-конкурентов в поколении (0 = выкл.)",
    "agent_generations": "поколений фабрики ИИ (синтез, развёртывание, дообучение)",
    "agent_heat_size": "ИИ-ботов в заезде арены (одна задача процесса)",
    "agent_heat_slots": "блоков, симулируемых в каждом заезде арены",
    "pattern_seed_bps": "доля ИИ-ботов поколения 0 из MEV-паттернов (bps, 0 = выкл.)",
    "pattern_limit": "самых тяжёлых MEV-паттернов, загружаемых в фабрику ИИ",
    "pattern_synthetic": "сгенерированных эталонных паттернов, если хранилище не задано",
    "custom_bots": "чертежей пользовательских ботов от ИИ в поколении (0 = выкл.)",
    "custom_generations": "поколений конструктора (проектирование, компиляция, оценка, дообучение)",
}

_FIELD = string.Formatter()


def fields_of(template: str) -> List[str]:
    """Format fields a template uses (sorted, de-duplicated)."""
    return sorted({name for _, name, _, _ in _FIELD.parse(template) if name})


def t(key: str, lang: Optional[str] = None, **params: Any) -> str:
    """The text of ``key`` in ``lang`` (resolved by :func:`resolve_language`), formatted with ``params``.

    Unknown keys raise KeyError: a missing translation is a bug, not something to paper over.
    """
    entry = CATALOG[key]
    text = entry[resolve_language(lang)]
    return text.format(**params) if params else text


MODE_KEYS: Dict[str, str] = {"moderate": "moderate", "moderate_scan": "moderate", "brutal": "brutal",
                             "brutal_aggressive": "brutal", "max": "max", "max_brutal": "max"}


def _mode_key(mode: Any) -> str:
    return MODE_KEYS[str(getattr(mode, "value", mode))]


def mode_label(mode: Any, lang: Optional[str] = None) -> str:
    """Display name of a run mode (short name, enum value or RunMode member)."""
    return t(f"mode.{_mode_key(mode)}", lang)


def mode_note(mode: Any, lang: Optional[str] = None) -> str:
    return t(f"mode_note.{_mode_key(mode)}", lang)


def table_title(table: str, lang: Optional[str] = None) -> str:
    return t(f"table.{table}", lang)


def setting_help(attr: str, english: str, lang: Optional[str] = None) -> str:
    """Help text of a setting: the Russian entry, or the schema's English text."""
    if resolve_language(lang) == "ru":
        return SETTING_HELP_RU[attr]
    return english


def is_russian(text: str) -> bool:
    """True when ``text`` contains Cyrillic or is a token written the same way in both languages."""
    return bool(_CYRILLIC.search(text)) or text.strip() in SAME_IN_BOTH


def catalog_issues(catalog: Optional[Mapping[str, Mapping[str, str]]] = None) -> List[str]:
    """Every structural problem in a catalog (empty when it is complete and consistent)."""
    issues: List[str] = []
    for key, entry in sorted((catalog if catalog is not None else CATALOG).items()):
        missing = [lang for lang in LANGUAGES if not entry.get(lang)]
        if missing:
            issues.append(f"{key}: missing {', '.join(missing)}")
            continue
        if fields_of(entry["ru"]) != fields_of(entry["en"]):
            issues.append(f"{key}: fields differ ({fields_of(entry['ru'])} vs {fields_of(entry['en'])})")
        if not is_russian(entry["ru"]):
            issues.append(f"{key}: Russian text has no Cyrillic")
        extra = sorted(set(entry) - set(LANGUAGES))
        if extra:
            issues.append(f"{key}: unknown languages {extra}")
    return issues


def coverage(keys: Iterable[str], lang: str = "ru") -> float:
    """Share of ``keys`` translated into ``lang`` (1.0 = complete)."""
    ks = list(keys)
    if not ks:
        return 1.0
    return sum(1 for k in ks if CATALOG.get(k, {}).get(lang)) / len(ks)


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

NBSP = " "


def group_int(value: int, lang: Optional[str] = None) -> str:
    """Integer with thousands separators: ``1 234 567`` (ru, no-break space) or ``1,234,567`` (en)."""
    text = f"{abs(int(value)):,}"
    if resolve_language(lang) == "ru":
        text = text.replace(",", NBSP)
    return ("-" if int(value) < 0 else "") + text


def format_decimal(value: float, digits: int = 2, lang: Optional[str] = None) -> str:
    """Fixed-point number with a decimal comma in Russian."""
    text = f"{float(value):.{digits}f}"
    return text.replace(".", ",") if resolve_language(lang) == "ru" else text


def format_units(amount: int, decimals: int = 18, digits: int = 6, lang: Optional[str] = None) -> str:
    """Exact integer base units as a decimal string (no float rounding before the cut)."""
    amount = int(amount)
    sign = "-" if amount < 0 else ""
    whole, frac = divmod(abs(amount), 10**decimals)
    frac_text = f"{frac:0{decimals}d}"[:digits] if decimals else ""
    sep = "," if resolve_language(lang) == "ru" else "."
    head = group_int(whole, lang)
    return f"{sign}{head}{sep}{frac_text}" if frac_text else f"{sign}{head}"


def plural_ru(n: int, one: str, few: str, many: str) -> str:
    """Russian plural form for ``n``: 1 файл, 2 файла, 5 файлов, 11 файлов, 21 файл."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


__all__ = [
    "CATALOG",
    "DEFAULT_LANGUAGE",
    "LANGUAGES",
    "LANG_ENV_VAR",
    "MODE_KEYS",
    "NBSP",
    "SAME_IN_BOTH",
    "SETTING_HELP_RU",
    "catalog_issues",
    "coverage",
    "fields_of",
    "format_decimal",
    "format_units",
    "group_int",
    "is_russian",
    "mode_label",
    "mode_note",
    "normalize_language",
    "plural_ru",
    "resolve_language",
    "setting_help",
    "t",
    "table_title",
]
