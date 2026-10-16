"""Machine-specific SO101 paths; override with environment variables on other hosts."""

import os

# so101-nexus checkout (pinned commit 7bdb671), e.g. SO101_NEXUS_SOURCE=/root/so101-nexus on GPUServer.
NEXUS_SOURCE = os.environ.get('SO101_NEXUS_SOURCE', '/home/loser/Study/so101-nexus')
