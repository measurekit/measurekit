---
"@measurekit/knowledge-tracing": major
---

Remove the `skillWeight` option of `elo`: a skill ability's update uses the same update size as the global ability's, which is the hierarchical Elo model of Nižnan, Pelánek and Řihák (2015). The option was only varied by tests, and the model has no such weight.
