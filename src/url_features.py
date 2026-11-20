
from urllib.parse import urlsplit
import ipaddress


def extract_features(url):

    # Clean the input
    url = url.strip()

    # Add HTTPS if the user does not provide a scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed_url = urlsplit(url)

    domain = parsed_url.hostname or ""
    domain = domain.lower()

    # Split domain into parts
    domain_parts = domain.split(".") if domain else []

    # Count subdomains using our own consistent definition
    no_of_subdomain = max(len(domain_parts) - 2, 0)

    # URL and domain lengths
    url_length = len(url)
    domain_length = len(domain)

    # Check whether domain is an IP address
    is_domain_ip = 0

    try:
        ipaddress.ip_address(domain)
        is_domain_ip = 1
    except ValueError:
        pass

    # Extract TLD
    tld = domain_parts[-1] if len(domain_parts) > 1 else ""
    tld_length = len(tld)

    # Count letters and digits
    no_of_letters = sum(c.isalpha() for c in url)
    no_of_digits = sum(c.isdigit() for c in url)

    # Count special characters
    no_of_equals = url.count("=")
    no_of_qmark = url.count("?")
    no_of_ampersand = url.count("&")

    no_of_other_special = sum(
        not c.isalnum() and c not in "=?&"
        for c in url
    )

    # Detect basic obfuscation indicators
    has_obfuscation = int(
        "%" in url or
        "@" in url or
        "\\" in url
    )

    no_of_obfuscated = url.count("%")

    # Calculate ratios
    obfuscation_ratio = (
        no_of_obfuscated / url_length if url_length else 0
    )

    letter_ratio = (
        no_of_letters / url_length if url_length else 0
    )

    digit_ratio = (
        no_of_digits / url_length if url_length else 0
    )

    special_ratio = (
        no_of_other_special + no_of_equals +
        no_of_qmark + no_of_ampersand
    ) / url_length if url_length else 0

    # Check HTTPS
    is_https = int(parsed_url.scheme == "https")

    return {
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip,
        "TLDLength": tld_length,
        "NoOfSubDomain": no_of_subdomain,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": no_of_obfuscated,
        "ObfuscationRatio": obfuscation_ratio,
        "NoOfLettersInURL": no_of_letters,
        "LetterRatioInURL": letter_ratio,
        "NoOfDegitsInURL": no_of_digits,
        "DegitRatioInURL": digit_ratio,
        "NoOfEqualsInURL": no_of_equals,
        "NoOfQMarkInURL": no_of_qmark,
        "NoOfAmpersandInURL": no_of_ampersand,
        "NoOfOtherSpecialCharsInURL": no_of_other_special,
        "SpacialCharRatioInURL": special_ratio,
        "IsHTTPS": is_https
    }
