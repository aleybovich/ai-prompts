# Security and encryption

Nimbus Notes protects your data in transit and at rest.

## Encryption

All traffic between your devices and the Nimbus cloud is encrypted with TLS 1.3.
Notes stored on Nimbus servers are encrypted at rest with AES-256.

Pro and Team users can additionally turn on **end-to-end encryption (E2EE)** for
specific notebooks. When E2EE is on, notes in that notebook are encrypted with a
key derived from your passphrase before they leave your device, so Nimbus servers
only ever see ciphertext. The trade-off: if you forget the passphrase for an
end-to-end encrypted notebook, nobody — including Nimbus support — can recover
those notes. There is no password reset for E2EE notebooks.

End-to-end encrypted notebooks cannot be searched on the server, so OCR and
server-side search do not work inside them. Search still works locally on your
own devices.

## Sign-in and accounts

Everyone signs in with an email address and password. Two-factor authentication
(2FA) is available on all plans, using an authenticator app or a hardware
security key.

Team plans add **single sign-on (SSO)** with SAML, so employees sign in through
your identity provider (such as Okta or Entra ID). Team plans also support SCIM,
which automatically creates and deactivates Nimbus accounts as people join or
leave in your identity provider.

## Data location and deletion

You can choose whether your data is stored in the United States or the European
Union when you create your account. Deleted notes go to Trash and are recoverable
for 30 days, after which they are permanently erased. You can export or delete
all of your data at any time from account settings.
