//! The address policy for author-directed HTTP: a model endpoint comes from a
//! tenant's configuration, so unless the provider is a local-inference one
//! (catalog `class: local`) it must resolve to publicly routable addresses
//! only (`common.AssertURLSafe`, `isGlobalIP`). The check runs in the
//! resolver, so the connection goes to an address that passed it.

use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};

/// `isGlobalIP`: false for unspecified, loopback, multicast, link-local,
/// private, CGNAT, benchmarking, documentation and reserved ranges, and for
/// IPv6 transition addresses that embed one of those.
pub fn is_global(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_global_v4(v4),
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_global_v4(v4);
            }
            is_global_v6(v6)
        }
    }
}

fn is_global_v4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    if ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_multicast()
        || ip.is_link_local()
        || ip.is_private()
    {
        return false;
    }
    !(o[0] == 0
        || (o[0] == 100 && o[1] & 0xC0 == 64)
        || (o[0] == 192 && o[1] == 0 && (o[2] == 0 || o[2] == 2))
        || (o[0] == 198 && o[1] == 51 && o[2] == 100)
        || (o[0] == 203 && o[1] == 0 && o[2] == 113)
        || (o[0] == 198 && (o[1] == 18 || o[1] == 19))
        || o[0] >= 240)
}

fn is_global_v6(ip: Ipv6Addr) -> bool {
    let s = ip.segments();
    let b = ip.octets();
    if ip.is_unspecified() || ip.is_loopback() || ip.is_multicast() {
        return false;
    }
    // fe80::/10 link-local, fc00::/7 unique local.
    if s[0] & 0xffc0 == 0xfe80 || s[0] & 0xfe00 == 0xfc00 {
        return false;
    }
    // 2001:db8::/32 documentation, 100::/64 discard-only.
    if s[0] == 0x2001 && s[1] == 0x0db8 {
        return false;
    }
    if s[0] == 0x0100 && s[1] == 0 && s[2] == 0 && s[3] == 0 {
        return false;
    }
    let v4 = |a: u8, b2: u8, c: u8, d: u8| is_global_v4(Ipv4Addr::new(a, b2, c, d));
    // 6to4 2002::/16, NAT64 64:ff9b::/96, NAT64 local-use 64:ff9b:1::/48,
    // Teredo 2001::/32, IPv4-compatible ::a.b.c.d.
    if s[0] == 0x2002 {
        return v4(b[2], b[3], b[4], b[5]);
    }
    if s[0] == 0x0064 && s[1] == 0xff9b {
        if s[2] == 0x0001 {
            return false;
        }
        if s[2..6].iter().all(|x| *x == 0) {
            return v4(b[12], b[13], b[14], b[15]);
        }
    }
    if s[0] == 0x2001 && s[1] == 0 {
        return v4(b[4], b[5], b[6], b[7])
            && v4(b[12] ^ 0xff, b[13] ^ 0xff, b[14] ^ 0xff, b[15] ^ 0xff);
    }
    if s[..6].iter().all(|x| *x == 0) {
        return v4(b[12], b[13], b[14], b[15]);
    }
    true
}

/// A `ureq` resolver that refuses non-public addresses when `strict`.
pub struct Policy {
    pub strict: bool,
}

impl ureq::Resolver for Policy {
    fn resolve(&self, netloc: &str) -> io::Result<Vec<SocketAddr>> {
        let addrs: Vec<SocketAddr> = netloc.to_socket_addrs()?.collect();
        if addrs.is_empty() {
            return Err(io::Error::other(format!(
                "hostname {netloc:?} resolved to no addresses"
            )));
        }
        if self.strict {
            for a in &addrs {
                if !is_global(a.ip()) {
                    return Err(io::Error::other(format!(
                        "URL resolves to a non-public address ({}), which is not allowed",
                        a.ip()
                    )));
                }
            }
        }
        Ok(addrs)
    }
}

/// An HTTP agent that applies [`Policy`].
pub fn agent(strict: bool, timeout: std::time::Duration) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .resolver(Policy { strict })
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout(timeout)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(s: &str) -> bool {
        is_global(s.parse().unwrap())
    }

    #[test]
    fn public_and_private_addresses() {
        for public in ["8.8.8.8", "11.250.0.13", "93.184.216.34", "2606:4700::1111"] {
            assert!(g(public), "{public}");
        }
        for private in [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.5.4",
            "192.168.1.11",
            "169.254.169.254",
            "0.0.0.0",
            "100.64.0.1",
            "192.0.2.1",
            "198.51.100.7",
            "203.0.113.9",
            "198.18.0.1",
            "240.0.0.1",
            "224.0.0.1",
            "::1",
            "::",
            "fe80::1",
            "fd00::1",
            "2001:db8::1",
            "::ffff:127.0.0.1",
            "2002:7f00:1::1",
            "64:ff9b::7f00:1",
            "64:ff9b:1::1",
        ] {
            assert!(!g(private), "{private}");
        }
    }
}
