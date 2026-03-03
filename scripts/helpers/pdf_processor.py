import sys
try:
    import pypdf
    with open(sys.argv[1], 'rb') as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            print(page.extract_text())
except Exception as e:
    try:
        import PyPDF2
        with open(sys.argv[1], 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                print(page.extract_text())
    except Exception as e2:
        print("Error:", e, e2)
