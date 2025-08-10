## add packages

* dont create virtualenv 
```bash
poetry add package-name --dry-run
poetry lock 
```

* extra
```bash
poetry add xxx --extras  ccc
```

* github

```bash
poetry add git+https://github.com/xxx/vvv.git@dev
```

* update

```bash
poetry update 包名
```