from codefusion.parsing.identifiers import (
    CodeTokenizer,
    identifier_variants,
    looks_like_code_identifier,
    normalize_symbol,
    split_identifier,
)


def lower(xs):
    return [x.lower() for x in xs]


def test_camel_case():
    assert lower(split_identifier("getUserAuthenticationToken")) == ["get", "user", "authentication", "token"]


def test_snake_case():
    assert split_identifier("get_user_auth_token") == ["get", "user", "auth", "token"]


def test_screaming_snake_case():
    assert split_identifier("MAX_RETRIES") == ["MAX", "RETRIES"]


def test_acronym_then_word():
    assert split_identifier("HTTPRequest") == ["HTTP", "Request"]
    assert split_identifier("XMLHttpRequest") == ["XML", "Http", "Request"]


def test_oauth2_token_components():
    v = identifier_variants("OAuth2Token")
    assert {"OAuth2Token", "OAuth", "OAuth2", "Token"} <= v


def test_numbers_and_punctuation():
    assert "sha256" in identifier_variants("sha256_digest")
    assert split_identifier("user.token") == ["user", "token"]
    assert split_identifier("$scope") == ["scope"]


def test_original_identifier_preserved_by_tokenizer():
    tok = CodeTokenizer(stem_tokens=False, remove_stopwords=False)
    toks = tok.tokenize("def getUserToken(): pass")
    assert "getusertoken" in toks
    assert {"get", "user", "token"} <= set(toks)


def test_nl_query_matches_code_tokens_with_stemming():
    tok = CodeTokenizer(stem_tokens=True)
    q = set(tok.tokenize("validating users"))
    d = set(tok.tokenize("def validate_user(user): ..."))
    assert q & d, (q, d)


def test_stopwords_removed():
    tok = CodeTokenizer()
    assert "the" not in tok.tokenize("the self return value")


def test_normalize_symbol_is_convention_insensitive():
    assert normalize_symbol("validate_user") == normalize_symbol("validateUser") == normalize_symbol("ValidateUser")


def test_code_identifier_heuristic():
    assert looks_like_code_identifier("validate_user")
    assert looks_like_code_identifier("getUser")
    assert looks_like_code_identifier("MAX_RETRIES")
    assert not looks_like_code_identifier("authentication")
