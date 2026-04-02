from src.evaluation import _icd_string_match

def sanity_check(perturber):
    print("\n" + "="*60 + "\nSANITY CHECK")
    sample   = ("Patient had myocardial infarction. "
                "Aspirin and heparin administered.")
    entities = perturber.extract_medical_entities(sample)
    print(f"  Entities found: {len(entities)}")
    for e in entities: print(f"    [{e['label']}] {e['text']}")
    exact, prefix = _icd_string_match("I21.0", "I21.0")
    print(f"  ICD exact  I21.0==I21.0: {'PASS' if exact else 'FAIL'}")
    exact, prefix = _icd_string_match("I21.0", "I21.9")
    print(f"  ICD prefix I21.0~I21.9 : {'PASS' if prefix and not exact else 'FAIL'}")
    exact, prefix = _icd_string_match("I21.0", "J18.9")
    print(f"  ICD no-match I21.0/J18.9: {'PASS' if not exact and not prefix else 'FAIL'}")
    print("="*60 + "\n")

