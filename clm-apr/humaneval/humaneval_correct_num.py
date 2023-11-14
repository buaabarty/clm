HUMANEVAL_CORRECT = {
    'codet5-small': {
        'CODET5_BASE_CODEFORM_MASKFORM_NOCOMMENT': ['CONCATENATE', 'ADD', 'DIGIT_SUM'],
        'CODET5_BASE_CODEFORM_COMMENTFORM_NOCOMMENT': ['DIGIT_SUM']
    },
    'codet5-base': {
        'CODET5_BASE_CODEFORM_MASKFORM_NOCOMMENT': ['CONCATENATE', 'FILTER_BY_PREFIX', 'MAX_ELEMENT', 'ADD', 'STRING_TO_MD5'],
        'CODET5_BASE_CODEFORM_COMMENTFORM_NOCOMMENT': []
    },
    'codet5-large': {
        'CODET5_BASE_CODEFORM_MASKFORM_NOCOMMENT': ['STRING_SEQUENCE', 'FILTER_BY_PREFIX', 'PAIRS_SUM_TO_ZERO', 'WILL_IT_FLY', 'ROUNDED_AVG', 'MOVE_ONE_BALL'],
        'CODET5_BASE_CODEFORM_COMMENTFORM_NOCOMMENT': ['FILTER_BY_PLEFIX']
    },
    'codegen-350M': {
        'CODEGEN_COMPLETE_CODEFORM_NOCOMMENT': [
            'HAS_CLOSE_ELEMENTS', 'PARSE_NESTED_PARENS', 'FILTER_BY_SUBSTRING', 'STRING_XOR', 'GREATEST_COMMON_DIVISOR', 'STRING_SEQUENCE',
            'COUNT_DISTINCT_CHARACTERS', 'RESCALE_TO_UNIT', 'FILTER_INTEGERS', 'STRLEN', 'FLIP_CASE', 'CONCATENATE', 'FILTER_BY_PREFIX', 'GET_POSITIVE',
            'IS_PRIME', 'MAX_ELEMENT', 'INCRE_LIST', 'CHANGE_BASE', 'MEDIAN', 'IS_PALINDROME', 'ADD', 'WILL_IT_FLY', 'SORT_ARRAY', 'SKJKASDKD', 'BY_LENGTH',
            'FACTORIAL', 'SORT_ARRAY_BINARY', 'IS_SORTED', 'SIMPLIFY', 'STRING_TO_MD5'
        ],
        'CODEGEN_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT': [
            'HAS_CLOSE_ELEMENTS', 'PARSE_NESTED_PARENS', 'FILTER_BY_SUBSTRING', 'STRING_SEQUENCE', 'COUNT_DISTINCT_CHARACTERS', 'PARSE_MUSIC', 'FILTER_INTEGERS',
             'FILTER_BY_PREFIX', 'GET_POSITIVE', 'INCR_LIST', 'LARGEST_PRIME_FACTOR', 'CIRCULAR_SHIFT', 'WILL_IT_FLY', 'SORT_ARRAY', 'COUNT_UPPER', 
             'BY_LENGTH', 'FACTORIAL', 'IS_SORTED'
        ]
    },
    'codegen-2B': {
        'CODEGEN_COMPLETE_CODEFORM_NOCOMMENT': [
            'HAS_CLOSE_ELEMENTS', 'MEAN_ABSOLUTE_DEVIATION', 'INTERSPERSE', 'PARSE_NESTED_PARENS', 'FILTER_BY_SUBSTRING', 'STRING_XOR', 
            'GREATEST_COMMON_DIVISOR', 'COUNT_DISTINCT_CHARACTERS', 'PARSE_MUSIC', 'HOW_MANY_TIMES', 'SORT_NUMBERS', 'RESCALE_TO_UNIT', 
            'FILTER_INTEGERS', 'STRLEN', 'FACTORIZE', 'REMOVE_DUPLICATES', 'FLIP_CASE', 'CONCATENATE', 'FILTER_BY_PREFIX', 'GET_POSITIVE', 
            'IS_PRIME', 'SORT_THIRD', 'MAX_ELEMENT', 'TRIPLES_SUM_TO_ZERO', 'INCR_LIST', 'PAIRS_SUM_TO_ZERO', 'CHANGE_BASE', 'MEDIAN', 
            'IS_PALINDROME', 'BELOW_THRESHOLD', 'ADD', 'FIB', 'CORRECT_BRACKETING', 'LARGEST_PRIME_FACTOR', 'SUM_TO_N', 'STRANGE_SORT_LIST', 
            'WILL_IT_FLY', 'PRIME_LENGTH', 'ANTI_SHUFFLE', 'SORT_ARRAY', 'SKJKASDKD', 'BY_LENGTH', 'FACTORIAL', 'EVEN_ODD_PALINDROME', 
            'HISTOGRAM', 'SORT_ARRAY_BINARY', 'IS_SORTED', 'SIMPLIFY', 'STRING_TO_MD5'
        ],
        'CODEGEN_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT': [
            'HAS_CLOSE_ELEMENTS', 'INTERSPERSE', 'PARSE_NESTED_PARENS', 'FILTER_BY_SUBSTRING', 'STRING_XOR', 'STRING_SEQUENCE', 
            'COUNT_DISTINCT_CHARACTERS', 'PARSE_MUSIC', 'HOW_MANY_TIMES', 'SORT_NUMBERS', 'FILTER_INTEGERS', 'STRLEN', 'FACTORIZE', 
            'REMOVE_DUPLICATES', 'FILTER_BY_PREFIX', 'GET_POSITIVE', 'IS_PRIME', 'TRIPLES_SUM_TO_ZERO', 'INCR_LIST', 'TRIANGLE_AREA', 
            'MODP', 'ADD', 'FIB', 'SUM_TO_N', 'CIRCULAR_SHIFT', 'WILL_IT_FLY', 'PRIME_LENGTH', 'GET_ROW', 'SORT_ARRAY', 'SKJKASDKD', 
            'COUNT_UPPER', 'BY_LENGTH', 'FACTORIAL', 'MAXIMUM_K', 'SOLUTION', 'IS_SORTED', 'X_OR_Y', 'STRING_TO_MD5'
        ]
    },
    'codegen-6B': {
        'CODEGEN_COMPLETE_CODEFORM_NOCOMMENT': [
            'HAS_CLOSE_ELEMENTS', 'BELOW_ZERO', 'INTERSPERSE', 'PARSE_NESTED_PARENS', 'FILTER_BY_SUBSTRING', 'ROLLING_MAX', 'STRING_XOR', 
            'GREATEST_COMMON_DIVISOR', 'ALL_PREFIXES', 'COUNT_DISTINCT_CHARACTERS', 'HOW_MANY_TIMES', 'RESCALE_TO_UNIT', 'FILTER_INTEGERS', 
            'FACTORIZE', 'REMOVE_DUPLICATES', 'FLIP_CASE', 'CONCATENATE', 'FILTER_BY_PREFIX', 'GET_POSITIVE', 'IS_PRIME', 'SORT_THIRD', 
            'MAX_ELEMENT', 'TRIPLES_SUM_TO_ZERO', 'INCR_LIST', 'PAIRS_SUM_TO_ZERO', 'CHANGE_BASE', 'MEDIAN', 'IS_PALINDROME', 'ADD', 
            'FIB', 'CORRECT_BRACKETING', 'LARGEST_PRIME_FACTOR', 'SUM_TO_N', 'WILL_IT_FLY', 'ANTI_SHUFFLE', 'SORT_ARRAY', 'SKJKASDKD', 
            'COUNT_UP_TO', 'CLOSEST_INTEGER', 'BY_LENGTH', 'FACTORIAL', 'EVEN_ODD_PALINDROME', 'HISTOGRAM', 'SORT_ARRAY_BINARY', 'IS_SORTED', 'SIMPLIFY'
        ],
        'CODEGEN_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT': [
            'HAS_CLOSE_ELEMENTS', 'INTERSPERSE', 'PARSE_NESTED_PARENS', 'FILTER_BY_SUBSTRING', 'ROLLING_MAX', 'ALL_PREFIXES', 'STRING_SEQUENCE', 
            'COUNT_DISTINCT_CHARACTERS', 'PARSE_MUSIC', 'HOW_MANY_TIMES', 'RESCALE_TO_UNIT', 'FILTER_INTEGERS', 'STRLEN', 'LARGEST_DIVISOR', 
            'FACTORIZE', 'CONCATENATE', 'FILTER_BY_PREFIX', 'GET_POSITIVE', 'IS_PRIME', 'MAX_ELEMENT', 'TRIPLES_SUM_TO_ZERO', 'INCR_LIST', 
            'PAIRS_SUM_TO_ZERO', 'CHANGE_BASE', 'TRIANGLE_AREA', 'MODP', 'REMOVE_VOWELS', 'ADD', 'SAME_CHARS', 'CORRECT_BRACKETING', 
            'LARGEST_PRIME_FACTOR', 'SUM_TO_N', 'CIRCULAR_SHIFT', 'WILL_IT_FLY', 'IS_HAPPY', 'PRIME_LENGTH', 'GET_ROW', 'SORT_ARRAY', 
            'SKJKASDKD', 'COUNT_UPPER', 'BY_LENGTH', 'FACTORIAL', 'HISTOGRAM', 'IS_SORTED', 'SIMPLIFY', 'SPECIAL_FILTER', 'RIGHT_ANGLE_TRIANGLE'
        ]
    },
    'plbart-base': {
        'PLBART_SEQFORM_MASKFORM_NOCOMMENT': [
            'SEPARATE_PAREN_GROUPS', 'BELOW_ZERO', 'STRING_XOR', 'LONGEST', 'COUNT_DISTINCT_CHARACTERS', 'FIND_CLOSEST_ELEMENTS', 
            'RESCALE_TO_UNIT', 'STRLEN', 'FLIP_CASE', 'CONCATENATE', 'FILTER_BY_PREFIX', 'FIND_ZERO', 'MAX_ELEMENT', 'TRIPLES_SUM_TO_ZERO', 
            'CAR_RACE_COLLISION', 'INCR_LIST', 'PAIRS_SUM_TO_ZERO', 'TRIANGLE_AREA', 'ADD', 'SAME_CHARS', 'COMMON', 'DIGIT_SUM', 'PLUCK', 
            'WILL_IT_FLY', 'TOTAL_MATCH', 'IS_MULTIPLY_PRIME', 'DECIMAL_TO_BINARY', 'ENCODE', 'CLOSEST_INTEGER', 'ROUNDED_AVG', 'BY_LENGTH', 
            'FACTORIAL', 'MOVE_ONE_BALL', 'EXCHANGE', 'HISTOGRAM', 'ODD_COUNT', 'MAX_FILL', 'VALID_DATE', 'STRING_TO_MD5'
        ],
        'PLBART_SEQFORM_COMMENTFORM_NOCOMMENT': [
            'TRUNCATE_NUMBER', 'COUNT_DISTINCT_CHARACTERS', 'STRLEN', 'CONCATENATE', 'FILTER_BY_PREFIX', 'CAR_RACE_COLLISION', 
            'INCR_LIST', 'ADD', 'MONOTONIC', 'DIGIT_SUM', 'CHECK_DICT_CASE', 'ROUNDED_AVG', 'FACTORIAL', 'HISTOGRAM', 'MIN_SUBARRAY_SUM', 
            'VALID_DATE', 'PROD_SIGNS', 'SIMPLIFY'
        ]
    },
    'plbart-large': {
        'PLBART_SEQFORM_MASKFORM_NOCOMMENT': [
            'SEPARATE_PAREN_GROUPS', 'TRUNCATE_NUMBER', 'MEAN_ABSOLUTE_DEVIATION', 'INTERSPERSE', 'PARSE_NESTED_PARENS', 'FILTER_BY_SUBSTRING', 
            'STRING_XOR', 'LONGEST', 'ALL_PREFIXES', 'STRING_SEQUENCE', 'COUNT_DISTINCT_CHARACTERS', 'SORT_NUMBERS', 'FIND_CLOSEST_ELEMENTS', 
            'RESCALE_TO_UNIT', 'FILTER_INTEGERS', 'STRLEN', 'FLIP_CASE', 'CONCATENATE', 'FILTER_BY_PREFIX', 'GET_POSITIVE', 'IS_PRIME', 
            'FIND_ZERO', 'UNIQUE', 'MAX_ELEMENT', 'TRIPLES_SUM_TO_ZERO', 'CAR_RACE_COLLISION', 'INCR_LIST', 'PAIRS_SUM_TO_ZERO', 'ADD', 
            'SAME_CHARS', 'MONOTONIC', 'COMMON', 'DIGIT_SUM', 'PLUCK', 'WILL_IT_FLY', 'TOTAL_MATCH', 'DECIMAL_TO_BINARY', 'NEXT_SMALLEST', 
            'ENCODE', 'COUNT_UP_TO', 'CLOSEST_INTEGER', 'ROUNDED_AVG', 'BY_LENGTH', 'FACTORIAL', 'COUNT_NUMS', 'MOVE_ONE_BALL', 'EXCHANGE', 
            'HISTOGRAM', 'MAX_FILL', 'GET_ODD_COLLATZ', 'VALID_DATE', 'STRING_TO_MD5'
        ],
        'PLBART_SEQFORM_COMMENTFORM_NOCOMMENT': [
            'TRUNCATE_NUMBER', 'FILTER_BY_SUBSTRING', 'HOW_MANY_TIMES', 'FIND_CLOSEST_ELEMENTS', 'RESCALE_TO_UNIT', 'FILTER_INTEGERS', 
            'STRLEN', 'LARGEST_DIVISOR', 'FLIP_CASE', 'CONCATENATE', 'GET_POSITIVE', 'IS_PRIME', 'FIND_ZERO', 'UNIQUE', 'MAX_ELEMENT', 
            'CAR_RACE_COLLISION', 'INCR_LIST', 'PAIRS_SUM_TO_ZERO', 'TRIANGLE_AREA', 'ADD', 'MONOTONIC', 'DIGIT_SUM', 'WILL_IT_FLY', 
            'TOTAL_MATCH', 'DECIMAL_TO_BINARY', 'IS_HAPPY', 'PRIME_LENGTH', 'CHECK_DICT_CASE', 'ROUNDED_AVG', 'FACTORIAL', 'HISTOGRAM', 
            'REVERSE_DELETE', 'MIN_SUBARRAY_SUM', 'VALID_DATE', 'IS_SORTED', 'PROD_SIGNS', 'SIMPLIFY', 'STRING_TO_MD5'
        ]
    },
    'codet5-small-finetune': [
        'SEPARATE_PAREN_GROUPS', 'TRUNCATE_NUMBER', 'MEAN_ABSOLUTE_DEVIATION', 'INTERSPERSE', 'ALL_PREFIXES', 'STRING_SEQUENCE', 
        'COUNT_DISTINCT_CHARACTERS', 'HOW_MANY_TIMES', 'FIND_CLOSEST_ELEMENTS', 'STRLEN', 'LARGEST_DIVISOR', 'FILTER_BY_PREFIX', 'IS_PRIME', 
        'MAX_ELEMENT', 'INCR_LIST', 'PAIRS_SUM_TO_ZERO', 'TRIANGLE_AREA', 'ADD', 'MONOTONIC', 'SUM_TO_N', 'TOTAL_MATCH', 'SOLVE', 'GET_ROW', 
        'CHECK_DICT_CASE', 'MAKE_A_PILE', 'ROUNDED_AVG', 'UNIQUE_DIGITS', 'BY_LENGTH', 'FACTORIAL', 'COUNT_NUMS', 'MOVE_ONE_BALL', 'EXCHANGE', 
        'HISTOGRAM', 'REVERSE_DELETE', 'ODD_COUNT', 'MAX_FILL', 'VALID_DATE', 'IS_SORTED', 'INTERSECTION', 'PROD_SIGNS', 'SIMPLIFY'
    ],
    'codet5-base-finetune': [
        'HAS_CLOSE_ELEMENTS', 'SEPARATE_PAREN_GROUPS', 'TRUNCATE_NUMBER', 'MEAN_ABSOLUTE_DEVIATION', 'INTERSPERSE', 'FILTER_BY_SUBSTRING', 
        'LONGEST', 'ALL_PREFIXES', 'STRING_SEQUENCE', 'COUNT_DISTINCT_CHARACTERS', 'HOW_MANY_TIMES', 'FIND_CLOSEST_ELEMENTS', 'RESCALE_TO_UNIT', 
        'STRLEN', 'LARGEST_DIVISOR', 'CONCATENATE', 'FILTER_BY_PREFIX', 'IS_PRIME', 'FIND_ZERO', 'UNIQUE', 'MAX_ELEMENT', 'TRIPLES_SUM_TO_ZERO', 
        'INCR_LIST', 'PAIRS_SUM_TO_ZERO', 'TRIANGLE_AREA', 'ADD', 'SAME_CHARS', 'MONOTONIC', 'DIGIT_SUM', 'TOTAL_MATCH', 'SOLVE', 'GET_ROW', 
        'NEXT_SMALLEST', 'CHECK_DICT_CASE', 'CHOOSE_NUM', 'ROUNDED_AVG', 'UNIQUE_DIGITS', 'BY_LENGTH', 'FACTORIAL', 'COUNT_NUMS', 
        'MOVE_ONE_BALL', 'EXCHANGE', 'HISTOGRAM', 'REVERSE_DELETE', 'ODD_COUNT', 'MIN_SUBARRAY_SUM', 'MAX_FILL', 'SOLUTION', 'VALID_DATE', 
        'IS_SORTED', 'INTERSECTION', 'PROD_SIGNS', 'SIMPLIFY', 'STRING_TO_MD5'
    ],
    'codet5-large-finetune': [
        'SEPARATE_PAREN_GROUPS', 'BELOW_ZERO', 'MEAN_ABSOLUTE_DEVIATION', 'INTERSPERSE', 'FILTER_BY_SUBSTRING', 'LONGEST', 'ALL_PREFIXES', 
        'STRING_SEQUENCE', 'COUNT_DISTINCT_CHARACTERS', 'HOW_MANY_TIMES', 'FIND_CLOSEST_ELEMENTS', 'RESCALE_TO_UNIT', 'FILTER_INTEGERS', 
        'STRLEN', 'LARGEST_DIVISOR', 'FILTER_BY_PREFIX', 'IS_PRIME', 'FIND_ZERO', 'MAX_ELEMENT', 'TRIPLES_SUM_TO_ZERO', 'INCR_LIST', 
        'PAIRS_SUM_TO_ZERO', 'ADD', 'SAME_CHARS', 'MONOTONIC', 'COMMON', 'DIGIT_SUM', 'WILL_IT_FLY', 'TOTAL_MATCH', 'SOLVE', 
        'ADD_EVEN_AT_ODD', 'GET_ROW', 'NEXT_SMALLEST', 'SKJKASDKD', 'CHOOSE_NUM', 'ROUNDED_AVG', 'BY_LENGTH', 'FACTORIAL', 
        'EVEN_ODD_PALINDROME', 'COUNT_NUMS', 'MOVE_ONE_BALL', 'EXCHANGE', 'HISTOGRAM', 'REVERSE_DELETE', 'ODD_COUNT', 'MIN_SUBARRAY_SUM', 
        'MAX_FILL', 'SOLUTION', 'GET_ODD_COLLATZ', 'VALID_DATE', 'IS_SORTED', 'INTERSECTION', 'COMPARE', 'STRING_TO_MD5'
    ],
    'codegen-350M-finetune': [
        'HAS_CLOSE_ELEMENTS', 'SEPARATE_PAREN_GROUPS', 'TRUNCATE_NUMBER', 'MEAN_ABSOLUTE_DEVIATION', 'INTERSPERSE', 'FILTER_BY_SUBSTRING', 
        'LONGEST', 'ALL_PREFIXES', 'STRING_SEQUENCE', 'HOW_MANY_TIMES', 'FILTER_INTEGERS', 'STRLEN', 'LARGEST_DIVISOR', 'REMOVE_DUPLICATES', 
        'FILTER_BY_PREFIX', 'GET_POSITIVE', 'UNIQUE', 'MAX_ELEMENT', 'TRIPLES_SUM_TO_ZERO', 'INCR_LIST', 'PAIRS_SUM_TO_ZERO', 'TRIANGLE_AREA',
         'MODP', 'ADD', 'SAME_CHARS', 'MONOTONIC', 'COMMON', 'SUM_TO_N', 'TOTAL_MATCH', 'DECIMAL_TO_BINARY', 'SOLVE', 'GET_ROW', 'NEXT_SMALLEST', 
         'COUNT_UPPER', 'ROUNDED_AVG', 'UNIQUE_DIGITS', 'BY_LENGTH', 'FACTORIAL', 'COUNT_NUMS', 'MOVE_ONE_BALL', 'EXCHANGE', 'HISTOGRAM', 
         'REVERSE_DELETE', 'ODD_COUNT', 'MAX_FILL', 'GET_CLOSET_VOWEL', 'SOLUTION', 'GET_ODD_COLLATZ', 'VALID_DATE', 'IS_SORTED', 'INTERSECTION', 
         'STRING_TO_MD5'
    ],
    'codegen-2B-finetune': [
        'SEPARATE_PAREN_GROUPS', 'TRUNCATE_NUMBER', 'MEAN_ABSOLUTE_DEVIATION', 'INTERSPERSE', 'FILTER_BY_SUBSTRING', 'LONGEST', 'ALL_PREFIXES', 
        'STRING_SEQUENCE', 'COUNT_DISTINCT_CHARACTERS', 'HOW_MANY_TIMES', 'FIND_CLOSEST_ELEMENTS', 'RESCALE_TO_UNIT', 'FILTER_INTEGERS', 
        'STRLEN', 'LARGEST_DIVISOR', 'FILTER_BY_PREFIX', 'GET_POSITIVE', 'MAX_ELEMENT', 'DECODE_CYCLIC', 'TRIPLES_SUM_TO_ZERO', 'INCR_LIST', 
        'PAIRS_SUM_TO_ZERO', 'TRIANGLE_AREA', 'ADD', 'SAME_CHARS', 'MONOTONIC', 'COMMON', 'SUM_TO_N', 'CIRCULAR_SHIFT', 'DIGIT_SUM', 
        'FRUIT_DISTRIBUTION', 'WILL_IT_FLY', 'TOTAL_MATCH', 'NEXT_SMALLEST', 'SKJKASDKD', 'CHECK_DICT_CASE', 'COUNT_UPPER', 'CLOSEST_INTEGER', 
        'ROUNDED_AVG', 'BY_LENGTH', 'COUNT_NUMS', 'MOVE_ONE_BALL', 'EXCHANGE', 'HISTOGRAM', 'REVERSE_DELETE', 'ODD_COUNT', 'MAX_FILL', 
        'SORT_ARRAY_BINARY', 'GET_CLOSET_VOWEL', 'SOLUTION', 'VALID_DATE', 'IS_SORTED', 'INTERSECTION'
    ],
    'codegen-6B-finetune': [
        'HAS_CLOSE_ELEMENTS', 'SEPARATE_PAREN_GROUPS', 'MEAN_ABSOLUTE_DEVIATION', 'INTERSPERSE', 'PARSE_NESTED_PARENS', 'FILTER_BY_SUBSTRING', 
        'LONGEST', 'ALL_PREFIXES', 'STRING_SEQUENCE', 'HOW_MANY_TIMES', 'FIND_CLOSEST_ELEMENTS', 'RESCALE_TO_UNIT', 'FILTER_INTEGERS', 
        'STRLEN', 'CONCATENATE', 'FILTER_BY_PREFIX', 'GET_POSITIVE', 'SORT_THIRD', 'MAX_ELEMENT', 'TRIPLES_SUM_TO_ZERO', 'INCR_LIST', 
        'PAIRS_SUM_TO_ZERO', 'TRIANGLE_AREA', 'BELOW_THRESHOLD', 'ADD', 'SAME_CHARS', 'MONOTONIC', 'COMMON', 'SUM_TO_N', 'DIGIT_SUM', 
        'TOTAL_MATCH', 'GET_ROW', 'NEXT_SMALLEST', 'CHECK_DICT_CASE', 'COUNT_UPPER', 'CHOOSE_NUM', 'ROUNDED_AVG', 'UNIQUE_DIGITS', 'BY_LENGTH', 
        'COUNT_NUMS', 'MOVE_ONE_BALL', 'EXCHANGE', 'HISTOGRAM', 'REVERSE_DELETE', 'ODD_COUNT', 'MAX_FILL', 'GET_CLOSET_VOWEL', 'VALID_DATE', 
        'IS_SORTED', 'INTERSECTION', 'PROD_SIGNS', 'STRING_TO_MD5'
    ],
    'plbart-base-finetune': [
        'INTERSPERSE', 'STRING_SEQUENCE', 'COUNT_DISTINCT_CHARACTERS', 'HOW_MANY_TIMES', 'FIND_CLOSEST_ELEMENTS', 'FILTER_INTEGERS', 'STRLEN', 
        'LARGEST_DIVISOR', 'FLIP_CASE', 'FILTER_BY_PREFIX', 'GET_POSITIVE', 'SORT_THIRD', 'MAX_ELEMENT', 'TRIPLES_SUM_TO_ZERO', 'PAIRS_SUM_TO_ZERO', 
        'ADD', 'SAME_CHARS', 'MONOTONIC', 'COMMON', 'SUM_TO_N', 'TOTAL_MATCH', 'ADD_EVEN_AT_ODD', 'GET_ROW', 'NEXT_SMALLEST', 'SKJKASDKD', 
        'CHOOSE_NUM', 'ROUNDED_AVG', 'UNIQUE_DIGITS', 'BY_LENGTH', 'FACTORIAL', 'COUNT_NUMS', 'MOVE_ONE_BALL', 'HISTOGRAM', 'REVERSE_DELETE', 
        'ODD_COUNT', 'MAX_FILL', 'GET_CLOSET_VOWEL', 'VALID_DATE', 'IS_SORTED', 'COMPARE_ONE', 'DOUBLE_THE_DIFFERENCE'
    ],
    'plbart-large-finetune': [
        'SEPARATE_PAREN_GROUPS', 'INTERSPERSE', 'LONGEST', 'ALL_PREFIXES', 'STRING_SEQUENCE', 'COUNT_DISTINCT_CHARACTERS', 'HOW_MANY_TIMES', 
        'FIND_CLOSEST_ELEMENTS', 'RESCALE_TO_UNIT', 'STRLEN', 'LARGEST_DIVISOR', 'CONCATENATE', 'FILTER_BY_PREFIX', 'IS_PRIME', 'SORT_THIRD', 
        'MAX_ELEMENT', 'TRIPLES_SUM_TO_ZERO', 'INCR_LIST', 'PAIRS_SUM_TO_ZERO', 'TRIANGLE_AREA', 'ADD', 'SAME_CHARS', 'MONOTONIC', 'COMMON', 
        'SUM_TO_N', 'DIGIT_SUM', 'TOTAL_MATCH', 'IS_MULTIPLY_PRIME', 'DECIMAL_TO_BINARY', 'ADD_EVEN_AT_ODD', 'GET_ROW', 'NEXT_SMALLEST', 
        'CHOOSE_NUM', 'ROUNDED_AVG', 'UNIQUE_DIGITS', 'BY_LENGTH', 'FACTORIAL', 'COUNT_NUMS', 'MOVE_ONE_BALL', 'EXCHANGE', 'HISTOGRAM', 
        'REVERSE_DELETE', 'ODD_COUNT', 'GET_CLOSET_VOWEL', 'VALID_DATE', 'IS_SORTED', 'SUM_SQUARES', 'STRING_TO_MD5'
    ],
    'CURE': [
        'BELOW_ZERO', 'INTERSPERSE', 'COUNT_DISTINCT_CHARACTERS', 'HOW_MANY_TIMES', 'FILTER_INTEGERS', 'STRLEN', 'FILTER_BY_PREFIX', 
        'PAIRS_SUM_TO_ZERO', 'CIRCULAR_SHIFT', 'WILL_IT_FLY', 'ENCODE', 'MAKE_A_PILE', 'CHOOSE_NUM', 'ROUNDED_AVG', 'BY_LENGTH', 
        'MOVE_ONE_BALL', 'REVERSE_DELETE', 'GET_CLOSET_VOWEL'
    ],
    'RewardRepair': [
        'HAS_CLOSE_ELEMENTS', 'INTERSPERSE', 'ALL_PREFIXES', 'HOW_MANY_TIMES', 'RESCALE_TO_UNIT', 'STRLEN', 'LARGEST_DIVISOR', 'CONCATENATE', 
        'FILTER_BY_PREFIX', 'MAX_ELEMENT', 'DECODE_CYCLIC', 'PAIRS_SUM_TO_ZERO', 'COMMON', 'CIRCULAR_SHIFT', 'IS_HAPPY', 'NEXT_SMALLEST', 
        'ROUNDED_AVG', 'BY_LENGTH', 'COUNT_NUMS', 'REVERSE_DELETE', 'SORT_ARRAY_BINARY', 'VALID_DATE'
    ],
    'Recoder': [
        'FILTER_BY_PREFIX', 'STRLEN', 'FIND_CLOSEST_ELEMENTS', 'PAIRS_SUM_TO_ZERO', 'MAKE_A_PILE', 'ADD', 'MIN_SUBARRAY_SUM', 'HISTOGRAM', 
        'HOW_MANY_TIMES', 'COUNT_UP_TO', 'GET_ROW'
    ],
    'Codex': [
        'HAS_CLOSE_ELEMENTS', 'SEPARATE_PAREN_GROUPS', 'BELOW_ZERO', 'MEAN_ABSOLUTE_DEVIATION', 'INTERSPERSE', 'PARSE_NESTED_PARENS', 
        'FILTER_BY_SUBSTRING', 'SUM_PRODUCT', 'ROLLING_MAX', 'STRING_XOR', 'LONGEST', 'GREATEST_COMMON_DIVISOR', 'ALL_PREFIXES', 
        'STRING_SEQUENCE', 'COUNT_DISTINCT_CHARACTERS', 'HOW_MANY_TIMES', 'SORT_NUMBERS', 'RESCALE_TO_UNIT', 'FILTER_INTEGERS', 'STRLEN', 
        'LARGEST_DIVISOR', 'FACTORIZE', 'REMOVE_DUPLICATES', 'FLIP_CASE', 'CONCATENATE', 'FILTER_BY_PREFIX', 'GET_POSITIVE', 'IS_PRIME', 
        'FIND_ZERO', 'SORT_THIRD', 'MAX_ELEMENT', 'TRIPLES_SUM_TO_ZERO', 'INCR_LIST', 'PAIRS_SUM_TO_ZERO', 'CHANGE_BASE', 'TRIANGLE_AREA', 
        'FIB4', 'MEDIAN', 'IS_PALINDROME', 'REMOVE_VOWELS', 'BELOW_THRESHOLD', 'ADD', 'FIB', 'CORRECT_BRACKETING', 'LARGEST_PRIME_FACTOR', 
        'SUM_TO_N', 'FIBFIB', 'CIRCULAR_SHIFT', 'STRANGE_SORT_LIST', 'WILL_IT_FLY', 'ISCUBE', 'PRIME_LENGTH', 'ADD_EVEN_AT_ODD', 'ANTI_SHUFFLE', 
        'SORT_ARRAY', 'SKJKASDKD', 'CHECK_DICT_CASE', 'COUNT_UP_TO', 'CLOSEST_INTEGER', 'WORDS_STRINGS', 'FACTORIAL', 'EVEN_ODD_PALINDROME', 
        'HISTOGRAM', 'SORT_ARRAY_BINARY', 'VALID_DATE', 'IS_SORTED', 'DIGITS', 'FIX_SPACES', 'SIMPLIFY', 'RIGHT_ANGLE_TRIANGLE', 'STRING_TO_MD5'
    ]
}

def get_humaneval_correct(model, config=None):
    if config is not None:
        return HUMANEVAL_CORRECT[model][config]
    return HUMANEVAL_CORRECT[model]

def print_correct_num():
    for model in (
        'codet5-small', 'codet5-base', 'codet5-large', 'codegen-350M', 'codegen-2B', 'codegen-6B', 
        'plbart-base', 'plbart-large',
        'codet5-small-finetune', 'codet5-base-finetune', 'codet5-large-finetune', 
        'codegen-350M-finetune', 'codegen-2B-finetune', 'codegen-6B-finetune',
        'plbart-base-finetune', 'plbart-large-finetune',
        'CURE', 'RewardRepair', 'Recoder', 'Codex'
    ):
        if type(HUMANEVAL_CORRECT[model]) == list:
            print(model, len(HUMANEVAL_CORRECT[model]))
        else:
            result = HUMANEVAL_CORRECT[model]
            for prompt in result:
                print(model, prompt, len(result[prompt]))


if __name__ == '__main__':
    print_correct_num()