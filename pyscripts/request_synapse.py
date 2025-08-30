import os
import bittensor as bt
import asyncio
from types import SimpleNamespace
from MIID.validator.reward import get_name_variation_rewards
from MIID.protocol import IdentitySynapse
import json
import hashlib
import sys
from MIID.neurons.miner.get_name_variations import get_name_variations_rewards
async def test_identity_synapse():
    # Initialize bittensor objects
    subtensor = bt.subtensor(network="finney")
    metagraph = subtensor.metagraph(54)  # Using netuid 91 as in original test
    wallet = bt.wallet(name="test", hotkey="miner")
    query_file = sys.argv[1] if len(sys.argv) > 1 else "/work/54/miners/pub54-2/net54_uid192/netuid54/miner/validator_59/run_2025-08-24-08-53/query.json"
    with open(query_file, 'r') as f:
        query_data = json.load(f)
    template = query_data['template']
    names = query_data['names']
    query_params = None
    if 'query_params' not in query_data:
        from neurons.miner.parse_query import query_parser
        query_params = asyncio.run(query_parser(template))
        query_data['query_params'] = query_params
    else:
        query_params = query_data['query_params']
    test_uid = 144 # 5H1jrSmC49vbTbXe8s68xBxHN6djqWQmpvEa8vTLCpfrUfJt
    coldkey = metagraph.axons[test_uid].coldkey
    try:
        async with bt.dendrite(wallet=wallet) as dendrite:
            # Create the synapse with sample data
            synapse = IdentitySynapse(
                names=names,
                query_template=template,
                timeout=720.0
            )

            # Test with a specific validator (using UID 101 as in original test)
            # test_uid = 19 # 5FvsYWZq6rwURnkscKYZfLmH7Emn7YacvGvX62XiX5WWgnGr
            axon = metagraph.axons[test_uid]
            
            bt.logging.info(f"Testing with validator UID={test_uid}, Hotkey={axon.hotkey}")
            
            synapse.dendrite.hotkey = "5C4qiYkqKjqGDSvzpf6YXCcnBgM6punh8BQJRP78bqMGsn54"
            # Send the query
            bt.logging.info(f"Sending query to validator UID={test_uid}, axon={axon}")
            response = await dendrite(
                axons=[axon],
                synapse=synapse,
                deserialize=True,  # We want the deserialized response
                timeout=720,  # Increased timeout for better reliability
            )
            # Process the response
            if response and len(response) > 0:
                try:
                    rewards, detailed_metrics = get_name_variations_rewards(
                        names, query_params, [response]
                    )
                except Exception as e:
                    bt.logging.error(f"Error during testing: {e}")
                variations = response[0]  # Get first response
                query_data['results'] = {
                    coldkey: {
                        "total_reward": rewards[0],
                        "variations": variations,
                        "metrics": detailed_metrics[0]
                    }
                }
                bt.logging.info(f"Received variations: {variations}")

                template_hash = hashlib.md5(template.encode()).hexdigest()
                workdir = f"tests/{test_uid}/{template_hash}"
                os.makedirs(workdir, exist_ok=True)
                with open(f"{workdir}/query.json", "w", encoding="utf-8") as f:
                    json.dump(query_data, f, indent=4)
                with open(f"{workdir}/variants.json", "w", encoding="utf-8") as f:
                    json.dump(variations, f, indent=4)
            else:
                bt.logging.error("No response received")

    except Exception as e:
        bt.logging.error(f"Error during testing: {e}")

if __name__ == "__main__":
    asyncio.run(test_identity_synapse())
    sys.exit(0)
