# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly.

### How to Report

**Please do NOT** report security vulnerabilities through public GitHub issues.

Instead, please report them via email to:

- **Email**: izzat.zaky@gmail.com
- **Subject**: `[SECURITY] Vulnerability Report`

### What to Include

Please include the following information in your report:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if any)
- Your contact information (optional, but helpful for follow-up)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution**: Depends on severity and complexity

### Security Best Practices

When using this pipeline with sensitive medical data (e.g., ADNI data):

1. **Follow ADNI Data Use Agreement** requirements strictly
2. **Do not upload data** to public-facing AI tools or cloud services without proper safeguards
3. **Use secure environments** for data processing
4. **Keep dependencies updated** to avoid known vulnerabilities
5. **Review third-party tool licenses** (see [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md))

### Scope

This security policy applies to:

- The codebase and its dependencies
- Configuration files and documentation
- Data handling practices

**Note**: This repository does not host ADNI or any medical data. Security concerns related to data handling should be addressed according to ADNI Data Use Agreement requirements.

## Security Updates

Security updates will be:

- Released as patch versions (e.g., 0.1.1, 0.1.2)
- Documented in [docs/CHANGELOG.md](docs/CHANGELOG.md)
- Tagged with appropriate security labels

Thank you for helping keep this project secure!


