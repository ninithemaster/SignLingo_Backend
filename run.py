from app import app

def list_routes(app):
    import urllib
    output = []
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods))
        url = urllib.parse.unquote(str(rule))
        line = f"{rule.endpoint:30s} {methods:20s} {url}"
        output.append(line)
    
    print("\nRegistered Routes:")
    print("\n".join(sorted(output)))

# Call this after app is fully defined
if __name__ == '__main__':
    list_routes(app)
    app.run(debug=True, host="0.0.0.0")

