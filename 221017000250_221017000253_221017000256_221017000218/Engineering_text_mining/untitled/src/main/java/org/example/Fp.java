package org.example;

public class Fp {

    private String id;
    private String requirementDocumentId;
    private String level1Fp;

    private String level2Fp;

    private String fpName;

    private String description;

    private String keyword;

    private int pvWorst;

    private int pv;

    private int pvBest;

    private int ac;


    public Fp(String id, String requirementDocumentId, String level1Fp, String level2Fp, String fpName, String description, String keyword, int pvWorst, int pv, int pvBest, int ac) {
        this.id = id;
        this.requirementDocumentId = requirementDocumentId;
        this.level1Fp = level1Fp;
        this.level2Fp = level2Fp;
        this.fpName = fpName;
        this.description = description;
        this.keyword = keyword;
        this.pvWorst = pvWorst;
        this.pv = pv;
        this.pvBest = pvBest;
        this.ac = ac;
    }


    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getRequirementDocumentId() {
        return requirementDocumentId;
    }

    public void setRequirementDocumentId(String requirementDocumentId) {
        this.requirementDocumentId = requirementDocumentId;
    }

    public String getLevel1Fp() {
        return level1Fp;
    }

    public void setLevel1Fp(String level1Fp) {
        this.level1Fp = level1Fp;
    }

    public String getLevel2Fp() {
        return level2Fp;
    }

    public void setLevel2Fp(String level2Fp) {
        this.level2Fp = level2Fp;
    }

    public String getFpName() {
        return fpName;
    }

    public void setFpName(String fpName) {
        this.fpName = fpName;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getKeyword() {
        return keyword;
    }

    public void setKeyword(String keyword) {
        this.keyword = keyword;
    }

    public int getPvWorst() {
        return pvWorst;
    }

    public void setPvWorst(int pvWorst) {
        this.pvWorst = pvWorst;
    }

    public int getPv() {
        return pv;
    }

    public void setPv(int pv) {
        this.pv = pv;
    }

    public int getPvBest() {
        return pvBest;
    }

    public void setPvBest(int pvBest) {
        this.pvBest = pvBest;
    }

    public int getAc() {
        return ac;
    }

    public void setAc(int ac) {
        this.ac = ac;
    }
}
