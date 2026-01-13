"""
agents.py - Parser and Strategy Builder Agents

These are NOT LLMs. They are deterministic rule-based mappers.
The "Parser" extracts structured intent from English.
The "Builder" maps intent to KB IDs.

No guessing. No hallucination. If uncertain, they return low confidence.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ParsedIntent:
    """Structured intent extracted from user input."""
    raw_input: str
    strategy_name: Optional[str]
    dataset_name: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    confidence: float  # 0.0 to 1.0
    missing_fields: List[str]
    ambiguities: List[str]


@dataclass
class BoundStrategy:
    """Strategy bound to KB artifacts."""
    strategy_id: str
    dataset_id: str
    indicator_ids: List[str]
    parameters: Dict[str, Any]
    is_valid: bool
    validation_errors: List[str]


class ParserAgent:
    """
    Deterministic Parser Agent.
    Extracts structured intent from natural language.
    Uses pattern matching, not ML inference.
    """

    # Known strategy patterns
    STRATEGY_PATTERNS = {
        r'\bsuper\s*trend\b': 'SuperTrend',
        r'\bsupertrend\b': 'SuperTrend',
        r'\bmoving average crossover\b': 'MA Crossover',
        r'\bma crossover\b': 'MA Crossover',
        r'\brsi\s*(reversal|strategy)?\b': 'RSI Strategy',
        r'\bmacd\s*(crossover|strategy)?\b': 'MACD Strategy',
        r'\bbollinger\s*(band|bands)?\b': 'Bollinger Bands',
        r'\bbb\s*(mean\s*reversion|strategy)?\b': 'Bollinger Bands',
    }
    
    # Command type patterns
    COMMAND_PATTERNS = {
        r'\b(backtest|run|test|execute)\b': 'BACKTEST',
        r'\b(calculate|compute|show)\s+(sharpe|ratio)\b': 'METRIC_SHARPE',
        r'\b(calculate|compute|show)\s+(sortino)\b': 'METRIC_SORTINO',
        r'\b(calculate|compute|show)\s+(drawdown|max\s*drawdown|mdd)\b': 'METRIC_DRAWDOWN',
        r'\b(calculate|compute|show)\s+(returns?|cagr)\b': 'METRIC_RETURNS',
        r'\b(download|export|save)\s+(chart|report|results?)\b': 'DOWNLOAD',
        r'\b(list|show)\s+(strategies|available)\b': 'LIST_STRATEGIES',
        r'\b(help|what can you do)\b': 'HELP',
    }

    # Known dataset patterns
    DATASET_PATTERNS = {
        r'\bnifty\b': 'NIFTY',
        r'\bbanknifty\b': 'BANKNIFTY',
        r'\bbank nifty\b': 'BANKNIFTY',
        r'\bsensex\b': 'SENSEX',
    }

    # Date patterns
    DATE_PATTERNS = [
        r'from\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4})',
        r'(\d{4}-\d{2}-\d{2})',
        r'(\d{1,2}/\d{1,2}/\d{4})',
        r'january\s+(\d{1,2})(?:st|nd|rd|th)?\s*(?:to|through|-)?\s*(?:\w+\s+)?(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})',
    ]

    def parse(self, user_input: str) -> ParsedIntent:
        """
        Parse user input into structured intent.
        Returns confidence score. If < threshold, system should pause.
        """
        input_lower = user_input.lower()
        missing_fields = []
        ambiguities = []

        # Extract strategy
        strategy_name = None
        strategy_matches = set()  # Use set to avoid duplicates
        for pattern, name in self.STRATEGY_PATTERNS.items():
            if re.search(pattern, input_lower):
                strategy_matches.add(name)
        
        if len(strategy_matches) == 1:
            strategy_name = list(strategy_matches)[0]
        elif len(strategy_matches) > 1:
            ambiguities.append(f"Multiple strategies detected: {list(strategy_matches)}")
        else:
            missing_fields.append("strategy")

        # Extract dataset
        dataset_name = None
        dataset_matches = set()  # Use set to avoid duplicates
        for pattern, name in self.DATASET_PATTERNS.items():
            if re.search(pattern, input_lower):
                dataset_matches.add(name)

        if len(dataset_matches) == 1:
            dataset_name = list(dataset_matches)[0]
        elif len(dataset_matches) > 1:
            ambiguities.append(f"Multiple datasets detected: {list(dataset_matches)}")
        else:
            missing_fields.append("dataset/symbol")

        # Extract dates (improved parsing)
        start_date = None
        end_date = None
        
        # Pattern: "from X to Y" or "from X through Y"
        date_patterns = [
            # "from january 1st to january 31st 2026"
            r'from\s+(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?\s*(?:to|through|-)\s*(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})',
            # "january 1 to january 31 2026"
            r'(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?\s*(?:to|through|-)\s*(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})',
        ]
        
        month_map = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12',
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09',
            'oct': '10', 'nov': '11', 'dec': '12'
        }
        
        for pattern in date_patterns:
            match = re.search(pattern, input_lower)
            if match:
                start_month = month_map.get(match.group(1).lower(), '01')
                start_day = match.group(2).zfill(2)
                end_month = month_map.get(match.group(3).lower(), '01')
                end_day = match.group(4).zfill(2)
                year = match.group(5)
                start_date = f"{year}-{start_month}-{start_day}"
                end_date = f"{year}-{end_month}-{end_day}"
                break
        
        # Don't mark missing if no date - we'll use defaults from sidebar

        # Calculate confidence
        confidence = 1.0
        confidence -= len(missing_fields) * 0.25
        confidence -= len(ambiguities) * 0.3
        confidence = max(0.0, confidence)

        return ParsedIntent(
            raw_input=user_input,
            strategy_name=strategy_name,
            dataset_name=dataset_name,
            start_date=start_date,
            end_date=end_date,
            confidence=confidence,
            missing_fields=missing_fields,
            ambiguities=ambiguities
        )


class StrategyBuilderAgent:
    """
    Maps parsed intent to KB artifact IDs.
    Does NOT create logic. Only references existing KB entries.
    """

    # Mapping from friendly names to KB IDs
    STRATEGY_MAP = {
        'SuperTrend': 'strat_supertrend_001',
        'MA Crossover': 'strat_ma_crossover_001',
        'RSI Strategy': 'strat_rsi_001',
        'MACD Strategy': 'strat_macd_001',
        'Bollinger Bands': 'strat_bollinger_001',
    }

    DATASET_MAP = {
        'NIFTY': 'dataset_nifty_fut_001',
        'BANKNIFTY': 'dataset_banknifty_fut_001',
        'SENSEX': 'dataset_sensex_001',
    }

    def __init__(self, kb):
        """Initialize with a KnowledgeBase instance."""
        self.kb = kb

    def build(self, intent: ParsedIntent) -> BoundStrategy:
        """
        Bind parsed intent to KB artifact IDs.
        Returns validation errors if artifacts don't exist.
        """
        validation_errors = []

        # Map strategy
        strategy_id = self.STRATEGY_MAP.get(intent.strategy_name)
        if not strategy_id:
            validation_errors.append(f"Unknown strategy: {intent.strategy_name}")
            strategy_id = ""
        else:
            # Verify it exists in KB
            if not self.kb.get_strategy(strategy_id):
                validation_errors.append(f"Strategy not found in KB: {strategy_id}")

        # Map dataset
        dataset_id = self.DATASET_MAP.get(intent.dataset_name)
        if not dataset_id:
            validation_errors.append(f"Unknown dataset: {intent.dataset_name}")
            dataset_id = ""
        else:
            # Verify it exists in KB
            if not self.kb.get_dataset(dataset_id):
                validation_errors.append(f"Dataset not found in KB: {dataset_id}")

        # Get required indicators from strategy
        indicator_ids = []
        if strategy_id:
            strategy = self.kb.get_strategy(strategy_id)
            if strategy:
                indicator_ids = strategy.get('required_indicators', [])
                # Verify indicators exist
                for ind_id in indicator_ids:
                    if not self.kb.get_indicator(ind_id):
                        validation_errors.append(f"Indicator not found in KB: {ind_id}")

        # Get parameters
        parameters = {}
        if strategy_id:
            strategy = self.kb.get_strategy(strategy_id)
            if strategy:
                parameters = strategy.get('parameters', {})

        is_valid = len(validation_errors) == 0

        return BoundStrategy(
            strategy_id=strategy_id,
            dataset_id=dataset_id,
            indicator_ids=indicator_ids,
            parameters=parameters,
            is_valid=is_valid,
            validation_errors=validation_errors
        )


class ValidationAgent:
    """
    Final gatekeeper before execution.
    Has VETO authority. Can reject execution if anything is wrong.
    """

    def __init__(self, kb):
        self.kb = kb

    def validate(self, bound_strategy: BoundStrategy) -> Tuple[bool, List[str]]:
        """
        Perform final validation.
        Returns (is_approved, list_of_issues)
        """
        issues = []

        # Check strategy exists and is valid
        if not bound_strategy.strategy_id:
            issues.append("VETO: No strategy specified")
        else:
            strategy = self.kb.get_strategy(bound_strategy.strategy_id)
            if not strategy:
                issues.append(f"VETO: Strategy {bound_strategy.strategy_id} not in KB")
            elif not self.kb.verify_hash(bound_strategy.strategy_id, 'strategy'):
                issues.append("VETO: Strategy hash verification failed")

        # Check dataset exists
        if not bound_strategy.dataset_id:
            issues.append("VETO: No dataset specified")
        else:
            dataset = self.kb.get_dataset(bound_strategy.dataset_id)
            if not dataset:
                issues.append(f"VETO: Dataset {bound_strategy.dataset_id} not in KB")

        # Check all required indicators exist
        for ind_id in bound_strategy.indicator_ids:
            if not self.kb.get_indicator(ind_id):
                issues.append(f"VETO: Required indicator {ind_id} not in KB")

        is_approved = len(issues) == 0
        return is_approved, issues
